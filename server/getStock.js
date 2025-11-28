const mongoose = require('mongoose');
const { calculateStartDate } = require('./getDate');

mongoose.connect('mongodb://127.0.0.1:27017/stocks', { useNewUrlParser: true, useUnifiedTopology: true });


const getAllNames = async (req, res) => {
  try {
      const database = mongoose.connection.db;

      // Get the list of collection names
      const collections = await database.listCollections().toArray();
      const collectionNames = collections
          .map(({ name }) => name)
          .filter(name => !name.endsWith("_predicted") && !name.endsWith("BoxJen"));

      res.json(collectionNames);
  } catch (error) {
      console.error(error);
      res.status(500).json({ error: 'Internal Server Error' });
  }
};



const getNames = async (req, res) => {
  try {
      const database = mongoose.connection.db;

      // Get the list of collection names
      const collections = await database.listCollections().toArray();
      const allNames = collections.map(({ name }) => name);

      // Filter out collections to ignore
      const filteredNames = allNames.filter(name => {
          return !name.endsWith("_predicted") && !name.endsWith("BoxJen");
      });

      // Get the search term from the request parameters
      const searchTerm = req.params.name;

      // Filter names based on the search term
      const matchingNames = filteredNames.filter(name => name.includes(searchTerm));

      res.json(matchingNames);

  } catch (error) {
      console.error(error);
      res.status(500).json({ error: 'Internal Server Error' });
  }
};



const getAllStockData = async (req, res) => {
  try {
    
    const database = mongoose.connection.db;
    // Get the list of collection names
    const collections = await database.listCollections().toArray();
    
    const filteredCollections = collections.filter(({ name }) => !name.endsWith("_predicted") && !name.endsWith("BoxJen"));

    // Fetch all documents from each collection
    const allStockData = await Promise.all(
      filteredCollections.map(async ({ name }) => {
        const collection = database.collection(name);
        const data = await collection.find({}).toArray(); // Adjust fields as needed
        return { collection: name, data };
      })
    );

    res.json(allStockData);
  } catch (error) {
    console.error(error);
    // Handle errors and send an appropriate response
    res.status(500).json({ error: 'Internal Server Error' });
  }
};



const getStockData = async (req, res) => {

    try {
        const database = mongoose.connection.db;
        const stockName = req.params.stockName;
        const collectionExists = await database.listCollections({ name: stockName }).hasNext();

        if (!collectionExists) {
            return res.status(404).json({ error: 'Stock not found' });
        }
        const collection = database.collection(stockName);
        const stockData = await collection.find({}).toArray(); // Adjust fields as needed

        res.json(stockData);
    } catch (error) {
        console.error(error);
        // Handle errors and send an appropriate response
        res.status(500).json({ error: 'Internal Server Error' });
    }
};


const getStockDataPeriod = async (req, res) => {
    try {
        const database = mongoose.connection.db;
        const stockName = req.params.stockName;
        const period = req.params.period;

        const collectionExists = await database.listCollections({ name: stockName }).hasNext();

        if (!collectionExists) {
            return res.status(404).json({ error: 'Stock not found' });
        }

        const collection = database.collection(stockName);

        let startDate;

        // Calculate the starting date based on the period
        switch (period.toLowerCase()) {
            case '1day':
                startDate = new Date();
                startDate.setDate(startDate.getDate()-2)
                break;
            case '1week':
                startDate = new Date();
                startDate.setDate(startDate.getDate() - 7);
                break;
            case '1month':
                startDate = new Date();
                startDate.setMonth(startDate.getMonth() - 1);
                break;
            case '1year':
                startDate = new Date();
                startDate.setFullYear(startDate.getFullYear() - 1);
                break;
            case '5years':
                startDate = new Date();
                startDate.setFullYear(startDate.getFullYear() - 5);
                break;
            default:
                return res.status(400).json({ error: 'Invalid period' });
        }

        // Query to find documents with an index value (date) greater than or equal to startDate
        const stockData = await collection.find({
            index: { $gte: new Date(startDate.toISOString()) }
        }).toArray();
        

        res.json( stockData );
    } catch (error) {
        console.error(error);
        // Handle errors and send an appropriate response
        res.status(500).json({ error: 'Internal Server Error' });
    }
};


const getPrediction = async (req, res) => {
    try {
      const database = mongoose.connection.db;
      const stockName = req.params.stockName;
      const period = req.params.period;
      
      let stockDataCollection;
      stockDataCollection = database.collection(`${stockName}_ARIMA_predicted`);
      const stockData = await stockDataCollection.find().toArray();
  
      // Sort the stockData in ascending order based on the key index
      //stockData.sort((a, b) => a.key - b.key);
  
      res.json(stockData);
    } catch (error) {
      console.error(error);
      // Handle errors and send an appropriate response
      res.status(500).json({ error: 'Internal Server Error' });
    }
  };
  

  const getPredictionLSTM = async (req, res) => {
    try {
      const database = mongoose.connection.db;
      const stockName = req.params.stockName;
      const period = req.params.period;
      
      let stockDataCollection;
      stockDataCollection = database.collection(`${stockName}_LSTM_predicted`);
      const stockData = await stockDataCollection.find().toArray();
  
      // Sort the stockData in ascending order based on the key index
      //stockData.sort((a, b) => a.key - b.key);
  
      res.json(stockData);
    } catch (error) {
      console.error(error);
      // Handle errors and send an appropriate response
      res.status(500).json({ error: 'Internal Server Error' });
    }
  };

  const getPredictionGCN = async (req, res) => {
    try {
      const database = mongoose.connection.db;
      const stockName = req.params.stockName;
      const period = req.params.period;
      
      let stockDataCollection;
      stockDataCollection = database.collection(`${stockName}_GCN_predicted`);
      const stockData = await stockDataCollection.find().toArray();
  
      // Sort the stockData in ascending order based on the key index
      //stockData.sort((a, b) => a.key - b.key);
  
      res.json(stockData);
    } catch (error) {
      console.error(error);
      // Handle errors and send an appropriate response
      res.status(500).json({ error: 'Internal Server Error' });
    }
  };

  const getPredictionBoxJenkins = async (req, res) => {
    try {
      const database = mongoose.connection.db;
      const stockName = req.params.stockName;
      const period = req.params.period;
      
      let stockDataCollection;
      stockDataCollection = database.collection(`${stockName}_predicted_BoxJen`);
      const stockData = await stockDataCollection.find().toArray();
  
      // Sort the stockData in ascending order based on the key index
      //stockData.sort((a, b) => a.key - b.key);
  
      res.json(stockData);
    } catch (error) {
      console.error(error);
      // Handle errors and send an appropriate response
      res.status(500).json({ error: 'Internal Server Error' });
    }
  };

module.exports = { getAllStockData,getStockData,getStockDataPeriod,getAllNames,getNames,getPrediction,getPredictionLSTM,getPredictionGCN,getPredictionBoxJenkins};
