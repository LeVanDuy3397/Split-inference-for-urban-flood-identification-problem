// app.js
const express = require('express');
const bodyParser = require('body-parser');
const amqp = require('amqplib');
const path = require('path');
const fs = require('fs');
const https = require('https');

const RABBITMQ_URL = process.env.RABBITMQ_URL || 'amqp://guest:guest@localhost:5672/';
const IMAGE_QUEUE = process.env.IMAGE_QUEUE || 'image_queue';
const LOCATION_QUEUE = process.env.LOCATION_QUEUE || 'location_queue';
const PREDICTION_AND_LOCATION_QUEUE = process.env.PREDICTION_AND_LOCATION_QUEUE || 'prediction_and_location';

const stored_prediction_and_location = [];
const app = express();
app.use(bodyParser.json({ limit: '25mb' })); // tăng giới hạn vì video frame nhiều
app.use(bodyParser.urlencoded({ extended: true, limit: '25mb' }));

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// optional: serve static assets if needed
app.use('/Css', express.static(path.join(__dirname, 'Css')));

let channel;

function addData(kinhdo, vido, timestamp, prediction) {
const newData = { prediction, kinhdo, vido, timestamp };
stored_prediction_and_location.push(newData);
}

// SSE clients
const sseClients = new Set();
function sseBroadcast(obj) {
const data = `data: ${JSON.stringify(obj)}\n\n`;
for (const res of sseClients) res.write(data);
}

app.get('/events', (req, res) => {
res.setHeader('Content-Type', 'text/event-stream');
res.setHeader('Cache-Control', 'no-cache');
res.setHeader('Connection', 'keep-alive');
res.flushHeaders?.();
res.write(': connected\n\n');
sseClients.add(res);
req.on('close', () => { sseClients.delete(res); });
});

async function initRabbit() {
try {
const conn = await amqp.connect(RABBITMQ_URL);
channel = await conn.createChannel();
await channel.assertQueue(IMAGE_QUEUE, { durable: true });
await channel.assertQueue(LOCATION_QUEUE, { durable: true });
await channel.assertQueue(PREDICTION_AND_LOCATION_QUEUE, { durable: true });
console.log('RabbitMQ ready, queues asserted:', IMAGE_QUEUE, LOCATION_QUEUE, PREDICTION_AND_LOCATION_QUEUE);// Consumer: nhận prediction + location từ Python
channel.consume(PREDICTION_AND_LOCATION_QUEUE, (msg) => {
  if (msg) {
    try {
      const data = JSON.parse(msg.content.toString('utf8'));
      const lon = Number(data.kinhdo);
      const lat = Number(data.vido);
      console.log('Message object:', data);
      console.log('kinhdo:', lon, 'vido:', lat);
      addData(data.kinhdo, data.vido, data.timestamp, data.prediction);
      channel.ack(msg);
    } catch (e) {
      console.error('Consumer parse error:', e, 'raw:', msg.content?.toString('utf8'));
      channel.ack(msg);
    }
  }
});
} catch (err) {
console.error('RabbitMQ init error:', err);
process.exit(1);
}
}
initRabbit();

// Trang chính
app.get('/', (req, res) => {
res.render('index');
});

// Nhận ảnh (bao gồm từng frame từ video) -> publish IMAGE_QUEUE
app.post('/api/send-image', async (req, res) => {
try {
const { image_base64, sent_at, from_video, frame_index, fps } = req.body;
if (!image_base64) {
return res.status(400).json({ ok: false, error: 'image_base64 is required' });
}
const msg = {
type: 'image',
image_base64,
sent_at: sent_at || Date.now(),
from_video: !!from_video,
frame_index: Number.isInteger(frame_index) ? frame_index : undefined,
fps: Number(fps) || undefined
};
channel.sendToQueue(IMAGE_QUEUE, Buffer.from(JSON.stringify(msg)), { persistent: true });
return res.json({ ok: true });
} catch (e) {
console.error('send-image error:', e);
return res.status(500).json({ ok: false, error: e.message });
}
});

// Tùy chọn: nhận meta của video (không bắt buộc cho pipeline)
app.post('/api/send-video-meta', async (req, res) => {
try {
const { filename, duration, fps, width, height } = req.body || {};
console.log('Video meta:', { filename, duration, fps, width, height });
return res.json({ ok: true });
} catch (e) {
return res.status(500).json({ ok: false, error: e.message });
}
});

// Nhận vị trí -> publish LOCATION_QUEUE
app.post('/api/send-location', async (req, res) => {
try {
const { lat, lng, timestamp } = req.body;
if (typeof lat !== 'number' || typeof lng !== 'number') {
return res.status(400).json({ ok: false, error: 'lat and lng must be numbers' });
}
const msg = { type: 'location', lat, lng, timestamp };
channel.sendToQueue(LOCATION_QUEUE, Buffer.from(JSON.stringify(msg)), { persistent: true });
return res.json({ ok: true });
} catch (e) {
console.error('send-location error:', e);
return res.status(500).json({ ok: false, error: e.message });
}
});

// Đếm số bản ghi đã lưu
app.get('/api/stored-count', (req, res) => {
res.json({ ok: true, count: stored_prediction_and_location.length });
});

// Phát dữ liệu đã lưu qua SSE
app.post('/api/broadcast-stored', async (req, res) => {
try {
if (stored_prediction_and_location.length === 0) {
return res.json({ ok: true, broadcasted: 0 });
}
const delayMs = Number(req.body?.delayMs) || 300;
for (const item of stored_prediction_and_location) {
const lon = Number(item.kinhdo);
const lat = Number(item.vido);
if (Number.isFinite(lat) && Number.isFinite(lon)) {
sseBroadcast({ lat, lng: lon, timestamp: item.timestamp, prediction: item.prediction });
if (delayMs > 0) await new Promise(r => setTimeout(r, delayMs));
}
}
return res.json({ ok: true, broadcasted: stored_prediction_and_location.length });
} catch (e) {
console.error('broadcast-stored error:', e);
return res.status(500).json({ ok: false, error: e.message });
}
});

const HTTP_PORT = process.env.PORT || 3000;
const HTTPS_PORT = process.env.HTTPS_PORT || 3001;

app.listen(HTTP_PORT, '0.0.0.0', () => {
console.log('HTTP at http://192.168.1.7:' + HTTP_PORT + ' run on computer');
});

// HTTPS self-signed
const sslOptions = {
key: fs.readFileSync(path.join(__dirname, 'certs', 'key.pem')),
cert: fs.readFileSync(path.join(__dirname, 'certs', 'cert.pem')),
};
https.createServer(sslOptions, app).listen(HTTPS_PORT, '0.0.0.0', () => {
console.log('HTTPS at https://192.168.1.7:' + HTTPS_PORT + ' run on telephone');
});