// controllers/view.controller.js
function getIndex(req, res) {
  res.render('index'); // index.ejs hiện đã tĩnh, không cần biến
}
module.exports = { getIndex };
