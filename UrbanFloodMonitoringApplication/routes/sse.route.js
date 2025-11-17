// routes/sse.route.js
const express = require('express');
const router = express.Router();
const { addClient, removeClient } = require('../services/sse');

router.get('/', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();
  res.write(': connected\n\n');
  addClient(res);
  req.on('close', () => removeClient(res));
});

module.exports = router;
