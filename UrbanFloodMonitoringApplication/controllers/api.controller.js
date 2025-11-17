// controllers/api.controller.js
const { getChannel } = require('../services/rabbit');
const { stored_prediction_and_location } = require('../services/storage');
const { broadcast } = require('../services/sse');

function sendImage(req, res) {
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
      fps: Number(fps) || undefined,
    };
    const channel = getChannel();
    const IMAGE_QUEUE = process.env.IMAGE_QUEUE || 'image_queue';
    channel.sendToQueue(IMAGE_QUEUE, Buffer.from(JSON.stringify(msg)), { persistent: true });
    return res.json({ ok: true });
  } catch (e) {
    console.error('send-image error:', e);
    return res.status(500).json({ ok: false, error: e.message });
  }
}

function sendVideoMeta(req, res) {
  try {
    const { filename, duration, fps, width, height } = req.body || {};
    console.log('Video meta:', { filename, duration, fps, width, height });
    return res.json({ ok: true });
  } catch (e) {
    return res.status(500).json({ ok: false, error: e.message });
  }
}

function sendLocation(req, res) {
  try {
    const { lat, lng, timestamp } = req.body;
    if (typeof lat !== 'number' || typeof lng !== 'number') {
      return res.status(400).json({ ok: false, error: 'lat and lng must be numbers' });
    }
    const msg = { type: 'location', lat, lng, timestamp };
    const channel = getChannel();
    const LOCATION_QUEUE = process.env.LOCATION_QUEUE || 'location_queue';
    channel.sendToQueue(LOCATION_QUEUE, Buffer.from(JSON.stringify(msg)), { persistent: true });
    return res.json({ ok: true });
  } catch (e) {
    console.error('send-location error:', e);
    return res.status(500).json({ ok: false, error: e.message });
  }
}

function getStoredCount(req, res) {
  res.json({ ok: true, count: stored_prediction_and_location.length });
}

async function broadcastStored(req, res) {
  try {
    const list = stored_prediction_and_location;
    if (list.length === 0) {
      return res.json({ ok: true, broadcasted: 0 });
    }
    const delayMs = Number(req.body?.delayMs) || 300;
    for (const item of list) {
      const lon = Number(item.kinhdo);
      const lat = Number(item.vido);
      if (Number.isFinite(lat) && Number.isFinite(lon)) {
        broadcast({ lat, lng: lon, timestamp: item.timestamp, prediction: item.prediction });
        if (delayMs > 0) await new Promise(r => setTimeout(r, delayMs));
      }
    }
    return res.json({ ok: true, broadcasted: list.length });
  } catch (e) {
    console.error('broadcast-stored error:', e);
    return res.status(500).json({ ok: false, error: e.message });
  }
}

module.exports = {
  sendImage,
  sendVideoMeta,
  sendLocation,
  getStoredCount,
  broadcastStored,
};
