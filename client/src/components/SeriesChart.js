import React, { useEffect, useRef, useState } from "react";
import { createChart } from "lightweight-charts";

const SeriesChart = ({ selectedName, selectedInterval, visibilityState, onToggleVisibility }) => {
  const chartContainerRef = useRef();
  const chartRef = useRef(null);
  const [seriesdata, setSeriesData] = useState([]);
  const [lineData, setLineData] = useState([]);
  const [lineDataLSTM, setLineDataLSTM] = useState([]);
  const [lineDataGCN, setLineDataGCN] = useState([]);
  const [lineDataBoxJenkins, setLineDataBoxJenkins] = useState([]);
  
  // Store series references
  const seriesRefs = useRef({});

  // Helper function to convert date to Unix timestamp
  const dateToTimestamp = (dateString) => {
    const date = new Date(dateString);
    if (isNaN(date.getTime())) {
      console.warn('Invalid date string:', dateString);
      return 0;
    }
    return Math.floor(date.getTime() / 1000);
  };

  // Helper function to validate and sort data
  const validateAndSortData = React.useCallback((data, dataType = 'line') => {
    if (!Array.isArray(data)) return [];
    
    const transformedData = data
      .map((item) => {
        const timestamp = dateToTimestamp(item.index);
        
        return {
          time: timestamp,
          value: typeof item.close === "number" ? item.close : 0
        };
      })
      .filter(item => item.time > 0 && !isNaN(item.time)) // Remove invalid timestamps
      .sort((a, b) => a.time - b.time); // Sort by timestamp ascending

    // Remove duplicates and keep only unique timestamps
    const uniqueData = [];
    const seenTimes = new Set();
    
    for (const item of transformedData) {
      if (!seenTimes.has(item.time)) {
        seenTimes.add(item.time);
        uniqueData.push(item);
      }
    }

    // Debug: Log data statistics
    console.log(`SeriesChart ${dataType} data processed:`, {
      original: data.length,
      filtered: transformedData.length,
      unique: uniqueData.length,
      timeRange: uniqueData.length > 0 ? {
        start: new Date(uniqueData[0].time * 1000).toISOString(),
        end: new Date(uniqueData[uniqueData.length - 1].time * 1000).toISOString()
      } : null
    });

    return uniqueData;
  }, []);

  const chartOptions = React.useMemo(() => ({
    layout: {
      textColor: "black",
      background: { type: "solid", color: "white" },
    },
    width: window.innerWidth,
    height: window.innerHeight - 100,
  }), []);

  // Fetch stock data (series)
  useEffect(() => {
    const fetchStockData = async () => {
      try {
        const response = await fetch(`http://localhost:8080/get-stock/${selectedName}/${selectedInterval}`);
        const rawData = await response.json();
        
        if (Array.isArray(rawData)) {
          const sortedData = validateAndSortData(rawData, 'stock');
          setSeriesData(sortedData);
        } else {
          console.error("Invalid stock data structure received:", rawData);
          setSeriesData([]);
        }
      } catch (error) {
        console.error("Error fetching stock data:", error);
        setSeriesData([]);
      }
    };

    fetchStockData();
  }, [selectedName, selectedInterval, validateAndSortData]);

  // Fetch prediction data
  useEffect(() => {
    const fetchPredictionData = async () => {
      try {
        const endpoints = [
          { url: `http://localhost:8080/get-prediction/${selectedName}`, setter: setLineData, name: 'ARIMA' },
          { url: `http://localhost:8080/get-prediction-LSTM/${selectedName}`, setter: setLineDataLSTM, name: 'LSTM' },
          { url: `http://localhost:8080/get-prediction-GCN/${selectedName}`, setter: setLineDataGCN, name: 'GCN' },
          { url: `http://localhost:8080/get-prediction-BoxJenkins/${selectedName}`, setter: setLineDataBoxJenkins, name: 'Box-Jenkins' }
        ];

        const promises = endpoints.map(async ({ url, setter, name }) => {
          try {
            const response = await fetch(url);
            const rawData = await response.json();
            
            if (Array.isArray(rawData)) {
              const sortedData = validateAndSortData(rawData, name);
              setter(sortedData);
            } else {
              console.error(`Invalid data structure from ${url}:`, rawData);
              setter([]);
            }
          } catch (error) {
            console.error(`Error fetching data from ${url}:`, error);
            setter([]);
          }
        });

        await Promise.all(promises);
      } catch (error) {
        console.error("Error in fetchPredictionData:", error);
      }
    };

    fetchPredictionData();
  }, [selectedName, validateAndSortData]); 

  // Create chart
  useEffect(() => {
    if (seriesdata.length === 0) {
      return;
    }

    // Clean up existing chart safely
    if (chartRef.current) {
      try {
        chartRef.current.remove();
      } catch (error) {
        console.warn("Chart removal error (safe to ignore):", error);
      }
      chartRef.current = null;
      seriesRefs.current = {};
    }

    const chart = createChart(chartContainerRef.current, chartOptions);
    chartRef.current = chart;

    // Create all series and store references
    const stockPriceSeries = chart.addLineSeries({ 
      color: "green", 
      lineWidth: 2,
      title: 'Stock Price'
    });
    const arimaSeries = chart.addLineSeries({ 
      color: 'orange', 
      lineWidth: 3,
      title: 'ARIMA'
    });
    const lstmSeries = chart.addLineSeries({ 
      color: 'blue', 
      lineWidth: 3,
      title: 'LSTM'
    });
    const gcnSeries = chart.addLineSeries({ 
      color: 'brown', 
      lineWidth: 3,
      title: 'GCN'
    });
    const boxjenkinsSeries = chart.addLineSeries({ 
      color: 'purple', 
      lineWidth: 3,
      title: 'Box-Jenkins'
    });

    // Store series references (only for prediction lines, not stock price)
    seriesRefs.current = {
      stockprice: stockPriceSeries,
      arima: arimaSeries,
      lstm: lstmSeries,
      gcn: gcnSeries,
      boxjenkins: boxjenkinsSeries
    };

    // Set data for each series with error handling
    const setSeriesData = (series, data, name) => {
      if (data && data.length > 0) {
        try {
          series.setData(data);
        } catch (error) {
          console.error(`Error setting ${name} data:`, error);
          console.log(`${name} data sample:`, data.slice(0, 5));
        }
      }
    };

    setSeriesData(stockPriceSeries, seriesdata, 'Stock Price');
    setSeriesData(arimaSeries, lineData, 'ARIMA');
    setSeriesData(lstmSeries, lineDataLSTM, 'LSTM');
    setSeriesData(gcnSeries, lineDataGCN, 'GCN');
    setSeriesData(boxjenkinsSeries, lineDataBoxJenkins, 'Box-Jenkins');

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        try {
          chart.applyOptions({ width: chartContainerRef.current.clientWidth });
        } catch (error) {
          console.warn("Resize error (chart may be disposed):", error);
        }
      }
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      if (chartRef.current) {
        try {
          chartRef.current.remove();
        } catch (error) {
          console.warn("Chart cleanup error (safe to ignore):", error);
        }
        chartRef.current = null;
        seriesRefs.current = {};
      }
    };
  }, [seriesdata, lineData, lineDataLSTM, lineDataGCN, lineDataBoxJenkins, chartOptions]);

  // Handle visibility changes
  useEffect(() => {
    if (!chartRef.current || !seriesRefs.current) return;

    try {
      Object.keys(visibilityState).forEach(key => {
        const series = seriesRefs.current[key];
        if (series) {
          series.applyOptions({
            visible: visibilityState[key]
          });
        }
      });
    } catch (error) {
      console.warn("Visibility update error (chart may be disposed):", error);
    }
  }, [visibilityState]);

  return (
    <div>
      <div ref={chartContainerRef} />
    </div>
  );
};

export default SeriesChart;
