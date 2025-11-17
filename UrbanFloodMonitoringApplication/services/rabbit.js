// services/rabbit.js
const amqp = require('amqplib');
const { addData } = require('./storage');

let channel;

async function initRabbit({ url, imageQueue, locationQueue, predAndLocQueue }) {
  const conn = await amqp.connect(url);
  channel = await conn.createChannel();
  await channel.assertQueue(imageQueue, { durable: true });
  await channel.assertQueue(locationQueue, { durable: true });
  await channel.assertQueue(predAndLocQueue, { durable: true });

  console.log('RabbitMQ ready, queues asserted:', imageQueue, locationQueue, predAndLocQueue);

  channel.consume(predAndLocQueue, (msg) => {
    if (!msg) return;
    try {
      const data = JSON.parse(msg.content.toString('utf8'));
      addData(data.kinhdo, data.vido, data.timestamp, data.prediction);
      // Lưu ý: nếu muốn broadcast realtime từ đây luôn:
      // const lon = Number(data.kinhdo);
      // const lat = Number(data.vido);
      // if (Number.isFinite(lat) && Number.isFinite(lon)) {
      //   const { broadcast } = require('./sse');
      //   broadcast({ lat, lng: lon, timestamp: data.timestamp, prediction: data.prediction });
      // }
      channel.ack(msg);
    } catch (e) {
      console.error('Consumer parse error:', e, 'raw:', msg.content?.toString('utf8'));
      channel.ack(msg);
    }
  });

  return channel;
}

function getChannel() {
  if (!channel) throw new Error('RabbitMQ channel not initialized yet');
  return channel;
}

module.exports = { initRabbit, getChannel };
