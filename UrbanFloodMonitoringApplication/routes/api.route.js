// routes/api.route.js
const express = require('express');
const router = express.Router();
const {
  sendImage,
  sendVideoMeta,
  sendLocation,
  getStoredCount,
  broadcastStored,
} = require('../controllers/api.controller');

router.post('/send-image', sendImage);
router.post('/send-video-meta', sendVideoMeta);
router.post('/send-location', sendLocation);
router.get('/stored-count', getStoredCount);
router.post('/broadcast-stored', broadcastStored);

module.exports = router;
