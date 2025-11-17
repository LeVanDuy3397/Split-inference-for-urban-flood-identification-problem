// routes/index.route.js
const express = require('express');
const router = express.Router();
const { getIndex } = require('../controllers/view.controller');

router.get('/', getIndex);

module.exports = router;
