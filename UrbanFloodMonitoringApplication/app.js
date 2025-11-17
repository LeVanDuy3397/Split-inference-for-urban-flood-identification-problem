// app.js
const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');

const app = express();

// Middlewares
app.use(bodyParser.json({ limit: '25mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '25mb' }));

// View engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Static
app.use(express.static(path.join(__dirname, 'public'))); 

// Routes
const indexRoute = require('./routes/index.route');
const apiRoute = require('./routes/api.route');
const sseRoute = require('./routes/sse.route');

app.use('/', indexRoute);
app.use('/api', apiRoute);
app.use('/events', sseRoute);

module.exports = app;
