// services/storage.js
const stored_prediction_and_location = [];

function addData(kinhdo, vido, timestamp, prediction) {
  const newData = { prediction, kinhdo, vido, timestamp };
  stored_prediction_and_location.push(newData);
}

module.exports = {
  stored_prediction_and_location,
  addData,
};
