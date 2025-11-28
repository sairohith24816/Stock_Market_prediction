import React, { useEffect, useRef, useState } from "react";
import { createChart } from "lightweight-charts";

const CandlestickChart = ({ selectedName, selectedInterval, visibilityState, onToggleVisibility }) => {
  const chartContainerRef = useRef();
  const chartRef = useRef(null);
  const [candlestickData, setCandlestickData] = useState([]);
  const [lineData, setLineData] = useState([]);
  const [lineDataLSTM, setLineDataLSTM] = useState([]);
  const [lineDataGCN, setLineDataGCN] = useState([]);
  const [lineDataBoxJenkins, setLineDataBoxJenkins] = useState([]);

  // Store series references
  const seriesRefs = useRef({});

  // Component cleanup on unmount
  useEffect(() => {
    return () => {
      if (chartRef.current) {
        try {
          chartRef.current.remove();
        } catch (error) {
          console.warn("Component unmount cleanup error (safe to ignore):", error);
        }
        chartRef.current = null;
        seriesRefs.current = {};
      }
    };
  }, []);

  // Helper function to convert date to Unix timestamp
  const dateToTimestamp = (dateString) => {
    const date = new Date(dateString);
    if (isNaN(date.getTime())) {
      console.warn('Invalid date string:', dateString);
      return 0;
    }
    return Math.floor(date.getTime() / 1000);
  };

  // Helper function to map selectedInterval label to milliseconds
  const intervalLabelToMs = (label) => {
    if (!label) return null;
    const l = String(label).trim().toLowerCase();
    if (l === '1d' || l.includes('1 day') || l.includes('1day') || l.includes('day')) return 24 * 60 * 60 * 1000;
    if (l === '1w' || l.includes('1 week') || l.includes('1week') || l.includes('week')) return 7 * 24 * 60 * 60 * 1000;
    if (l === '1m' || l.includes('1 month') || l.includes('1month') || l.includes('month')) return 30 * 24 * 60 * 60 * 1000; // approx 30 days
    if (l === '1y' || l.includes('1 year') || l.includes('1year') || l.includes('year')) return 365 * 24 * 60 * 60 * 1000; // approx 365 days
    if (l === '5y' || l.includes('5 years') || l.includes('5years')) return 5 * 365 * 24 * 60 * 60 * 1000; // approx 5 years
    // fallback: try parse numbers like '7d' or '30d'
    const numMatch = l.match(/(\d+)\s*(d|day|w|week|m|month|y|year)/);
    if (numMatch) {
      const n = parseInt(numMatch[1], 10);
      const unit = numMatch[2];
      if (unit.startsWith('d')) return n * 24 * 60 * 60 * 1000;
      if (unit.startsWith('w')) return n * 7 * 24 * 60 * 60 * 1000;
      if (unit.startsWith('m')) return n * 30 * 24 * 60 * 60 * 1000;
      if (unit.startsWith('y')) return n * 365 * 24 * 60 * 60 * 1000;
    }
    return null;
  };

  // Filter a dataset to only include the most recent intervalMs milliseconds (based on that dataset's own latest timestamp)
  const filterDataByInterval = (data, intervalMs) => {
    if (!Array.isArray(data) || data.length === 0 || !intervalMs) return data;
    // timestamps are in seconds
    const times = data.map(d => d.time).filter(t => typeof t === 'number' && !isNaN(t));
    if (times.length === 0) return data;
    const maxTimeSec = Math.max(...times);
    const cutoffSec = maxTimeSec - Math.floor(intervalMs / 1000);
    return data.filter(d => d.time >= cutoffSec);
  };

  // Helper function to validate and sort data
  const validateAndSortData = React.useCallback((data, dataType = 'line') => {
    if (!Array.isArray(data)) return [];

    const transformedData = data
      .map((item) => {
        const timestamp = dateToTimestamp(item.index);

        if (dataType === 'candlestick') {
          return {
            time: timestamp,
            open: typeof item.open === "number" ? item.open : 0,
            high: typeof item.high === "number" ? item.high : 0,
            low: typeof item.low === "number" ? item.low : 0,
            close: typeof item.close === "number" ? item.close : 0
          };
        } else {
          // Line data (predictions)
          return {
            time: timestamp,
            value: typeof item.close === "number" ? item.close : 0
          };
        }
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
    console.log(`${dataType} data processed:`, {
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

  // Fetch stock data (candlestick)
  useEffect(() => {
    const fetchStockData = async () => {
      try {
        const response = await fetch(`http://localhost:8080/get-stock/${selectedName}/${selectedInterval}`);
        const rawData = await response.json();

        if (Array.isArray(rawData)) {
          const sortedData = validateAndSortData(rawData, 'candlestick');
          const ms = intervalLabelToMs(selectedInterval);
          const filtered = filterDataByInterval(sortedData, ms);
          setCandlestickData(filtered);
        } else {
          console.error("Invalid candlestick data structure received:", rawData);
          setCandlestickData([]);
        }
      } catch (error) {
        console.error("Error fetching stock data:", error);
        setCandlestickData([]);
      }
    };

    fetchStockData();
  }, [selectedName, selectedInterval, validateAndSortData]);

  // Fetch prediction data
  useEffect(() => {
    const fetchPredictionData = async () => {
      try {
        const endpoints = [
          { url: `http://localhost:8080/get-prediction/${selectedName}`, setter: setLineData },
          { url: `http://localhost:8080/get-prediction-LSTM/${selectedName}`, setter: setLineDataLSTM },
          { url: `http://localhost:8080/get-prediction-GCN/${selectedName}`, setter: setLineDataGCN },
          { url: `http://localhost:8080/get-prediction-BoxJenkins/${selectedName}`, setter: setLineDataBoxJenkins }
        ];

        const ms = intervalLabelToMs(selectedInterval);

        const promises = endpoints.map(async ({ url, setter }) => {
          try {
            const response = await fetch(url);
            const rawData = await response.json();

            if (Array.isArray(rawData)) {
              const sortedData = validateAndSortData(rawData, 'line');
              const filtered = filterDataByInterval(sortedData, ms);
              setter(filtered);
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
  }, [selectedName, selectedInterval, validateAndSortData]);

  // Create chart
  useEffect(() => {
    if (candlestickData.length === 0) {
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

    chartRef.current = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth - 50,
      height: window.innerHeight - 230,
    });

    const candlestickSeries = chartRef.current.addCandlestickSeries({
      upColor: "rgba(0, 150, 36, 1)",
      downColor: "rgba(255, 0, 0, 1)",
      borderUpColor: "rgba(0, 150, 36, 1)",
      borderDownColor: "rgba(255, 0, 0, 1)",
      wickUpColor: "rgba(0, 150, 36, 1)",
      wickDownColor: "rgba(255, 0, 0, 1)",
    });

    // Set candlestick data with error handling
    try {
      candlestickSeries.setData(candlestickData);
    } catch (error) {
      console.error("Error setting candlestick data:", error);
      console.log("Candlestick data sample:", candlestickData.slice(0, 5));
    }

    // Create prediction series and store references
    const arimaSeries = chartRef.current.addLineSeries({
      color: 'orange',
      lineWidth: 2,
      title: 'ARIMA'
    });
    const lstmSeries = chartRef.current.addLineSeries({
      color: 'blue',
      lineWidth: 2,
      title: 'LSTM'
    });
    const gcnSeries = chartRef.current.addLineSeries({
      color: 'brown',
      lineWidth: 2,
      title: 'GCN'
    });
    const boxjenkinsSeries = chartRef.current.addLineSeries({
      color: 'purple',
      lineWidth: 2,
      title: 'Box-Jenkins'
    });

    // Store series references
    seriesRefs.current = {
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

    setSeriesData(arimaSeries, lineData, 'ARIMA');
    setSeriesData(lstmSeries, lineDataLSTM, 'LSTM');
    setSeriesData(gcnSeries, lineDataGCN, 'GCN');
    setSeriesData(boxjenkinsSeries, lineDataBoxJenkins, 'Box-Jenkins');

    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        try {
          chartRef.current.applyOptions({
            width: chartContainerRef.current.clientWidth - 50,
          });
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
  }, [candlestickData, lineData, lineDataLSTM, lineDataGCN, lineDataBoxJenkins]);

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

export default CandlestickChart;
