// services/sse.js
const sseClients = new Set();

function addClient(res) {
  sseClients.add(res);
}

function removeClient(res) {
  sseClients.delete(res);
}

function broadcast(obj) {
  const data = `data: ${JSON.stringify(obj)}\n\n`;
  for (const res of sseClients) {
    res.write(data);
  }
}

module.exports = { addClient, removeClient, broadcast };
