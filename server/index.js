const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const getStock = require('./getStock'); // Adjust the path to your actual file

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 8080;
app.use(bodyParser.json());

app.get("/", (req, res) => {
  res.send("Welcome to the Agentic AI Stock Market Prediction API!");
});

app.get("/get-names/:name", getStock.getNames);
app.get("/get-all-names", getStock.getAllNames);
app.get("/get-all-stock", getStock.getAllStockData);
app.get("/get-stock/:stockName", getStock.getStockData);
app.get("/get-stock/:stockName/:period", getStock.getStockDataPeriod);
app.get("/get-prediction/:stockName", getStock.getPrediction);
app.get("/get-prediction-LSTM/:stockName", getStock.getPredictionLSTM);
app.get("/get-prediction-GCN/:stockName", getStock.getPredictionGCN);
app.get("/get-prediction-BoxJenkins/:stockName", getStock.getPredictionBoxJenkins);


app.listen(PORT, () => {
  console.log(`Server On: http://localhost:${PORT}`);
});
