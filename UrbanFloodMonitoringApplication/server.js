// server.js
require('dotenv').config?.();

const fs = require('fs');
const https = require('https');
const path = require('path');

const app = require('./app');
const { initRabbit } = require('./services/rabbit');

const RABBITMQ_URL = process.env.RABBITMQ_URL || 'amqp://guest:guest@localhost:5672/';
const IMAGE_QUEUE = process.env.IMAGE_QUEUE || 'image_queue';
const LOCATION_QUEUE = process.env.LOCATION_QUEUE || 'location_queue';
const PREDICTION_AND_LOCATION_QUEUE = process.env.PREDICTION_AND_LOCATION_QUEUE || 'prediction_and_location';

const HTTP_PORT = process.env.PORT || 3000;
const HTTPS_PORT = process.env.HTTPS_PORT || 3001;

// Start HTTP
app.listen(HTTP_PORT, '0.0.0.0', () => {
  console.log('HTTP at http://192.168.1.7:' + HTTP_PORT + ' run on computer');
});

// Start HTTPS
const sslOptions = {
  key: fs.readFileSync(path.join(__dirname, 'certs', 'key.pem')),
  cert: fs.readFileSync(path.join(__dirname, 'certs', 'cert.pem')),
};

https.createServer(sslOptions, app).listen(HTTPS_PORT, '0.0.0.0', () => {
  console.log('HTTPS at https://192.168.1.7:' + HTTPS_PORT + ' run on telephone');
});

// Init RabbitMQ
initRabbit({
  url: RABBITMQ_URL,
  imageQueue: IMAGE_QUEUE,
  locationQueue: LOCATION_QUEUE,
  predAndLocQueue: PREDICTION_AND_LOCATION_QUEUE,
}).catch(err => {
  console.error('RabbitMQ init error:', err);
  process.exit(1);
});
