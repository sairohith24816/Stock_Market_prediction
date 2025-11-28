import React, { useEffect, useRef, useState } from "react";
import { createChart } from "lightweight-charts";

const CandlestickChart = ({ selectedName, selectedInterval, visibilityState, onToggleVisibility }) => {
  const chartContainerRef = useRef();
  const chartRef = useRef(null);
  const legendRef = useRef(null);
  const [candlestickData, setCandlestickData] = useState([]);
  const [lineData, setLineData] = useState([]);
  const [lineDataLSTM, setLineDataLSTM] = useState([]);
  const [lineDataGCN, setLineDataGCN] = useState([]);
  const [lineDataBoxJenkins, setLineDataBoxJenkins] = useState([]);
  const [legendData, setLegendData] = useState({
    time: '',
    open: '',
    high: '',
    low: '',
    close: '',
    arima: '',
    lstm: '',
    gcn: ''
  });

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
          setCandlestickData(sortedData);
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

  // Helper function to get start timestamp based on interval
  const getStartDateTimestamp = (interval) => {
    const now = new Date();
    switch (interval.toLowerCase()) {
      case '1day':
        now.setDate(now.getDate() - 2);
        break;
      case '1week':
        now.setDate(now.getDate() - 7);
        break;
      case '1month':
        now.setMonth(now.getMonth() - 1);
        break;
      case '1year':
        now.setFullYear(now.getFullYear() - 1);
        break;
      case '5years':
        now.setFullYear(now.getFullYear() - 5);
        break;
      default:
        now.setFullYear(now.getFullYear() - 1);
    }
    return Math.floor(now.getTime() / 1000);
  };

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

        const promises = endpoints.map(async ({ url, setter }) => {
          try {
            const response = await fetch(url);
            const rawData = await response.json();

            if (Array.isArray(rawData)) {
              let sortedData = validateAndSortData(rawData, 'line');
              
              // Filter data based on selected interval
              const startTime = getStartDateTimestamp(selectedInterval);
              sortedData = sortedData.filter(item => item.time >= startTime);
              
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
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        mouseWheel: true,
        pinch: true,
        axisPressedMouseMove: {
          time: true,
          price: true,
        },
        axisDoubleClickReset: {
          time: true,
          price: true,
        },
      },
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
      title: 'ARIMA',
      visible: visibilityState.arima
    });
    const lstmSeries = chartRef.current.addLineSeries({
      color: 'blue',
      lineWidth: 2,
      title: 'LSTM',
      visible: visibilityState.lstm
    });
    const gcnSeries = chartRef.current.addLineSeries({
      color: 'brown',
      lineWidth: 2,
      title: 'GCN',
      visible: visibilityState.gcn
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

    // Fit the chart to show only the data within the selected interval
    chartRef.current.timeScale().fitContent();

    // Subscribe to crosshair move events
    chartRef.current.subscribeCrosshairMove((param) => {
      if (!param || !param.time || !param.point) {
        setLegendData({
          time: '',
          open: '',
          high: '',
          low: '',
          close: '',
          arima: '',
          lstm: '',
          gcn: ''
        });
        return;
      }

      const dateStr = new Date(param.time * 1000).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      });

      const candleData = param.seriesData.get(candlestickSeries);
      const arimaValue = param.seriesData.get(seriesRefs.current.arima);
      const lstmValue = param.seriesData.get(seriesRefs.current.lstm);
      const gcnValue = param.seriesData.get(seriesRefs.current.gcn);

      setLegendData({
        time: dateStr,
        open: candleData?.open ? candleData.open.toFixed(2) : '',
        high: candleData?.high ? candleData.high.toFixed(2) : '',
        low: candleData?.low ? candleData.low.toFixed(2) : '',
        close: candleData?.close ? candleData.close.toFixed(2) : '',
        arima: arimaValue?.value ? arimaValue.value.toFixed(2) : '',
        lstm: lstmValue?.value ? lstmValue.value.toFixed(2) : '',
        gcn: gcnValue?.value ? gcnValue.value.toFixed(2) : ''
      });
    });

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
    <div style={{ position: 'relative' }}>
      <div ref={chartContainerRef} />
      {legendData.time && (
        <div
          ref={legendRef}
          style={{
            position: 'absolute',
            top: '10px',
            left: '10px',
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            border: '1px solid #ccc',
            borderRadius: '6px',
            padding: '10px',
            fontSize: '12px',
            fontFamily: 'Arial, sans-serif',
            boxShadow: '0 2px 6px rgba(0, 0, 0, 0.15)',
            zIndex: 1000,
            pointerEvents: 'none'
          }}
        >
          <div style={{ fontWeight: 'bold', marginBottom: '6px' }}>{legendData.time}</div>
          {legendData.open && (
            <div style={{ marginBottom: '4px' }}>
              <span style={{ fontWeight: '500' }}>O: </span>
              <span style={{ color: '#333' }}>{legendData.open}</span>
              <span style={{ fontWeight: '500', marginLeft: '8px' }}>H: </span>
              <span style={{ color: '#333' }}>{legendData.high}</span>
              <span style={{ fontWeight: '500', marginLeft: '8px' }}>L: </span>
              <span style={{ color: '#333' }}>{legendData.low}</span>
              <span style={{ fontWeight: '500', marginLeft: '8px' }}>C: </span>
              <span style={{ color: '#333' }}>{legendData.close}</span>
            </div>
          )}
          {legendData.arima && visibilityState.arima && (
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
              <span style={{ width: '10px', height: '10px', backgroundColor: 'orange', borderRadius: '50%', marginRight: '6px' }}></span>
              <span style={{ fontWeight: '500' }}>ARIMA: </span>
              <span style={{ marginLeft: '4px', color: 'orange' }}>{legendData.arima}</span>
            </div>
          )}
          {legendData.lstm && visibilityState.lstm && (
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
              <span style={{ width: '10px', height: '10px', backgroundColor: 'blue', borderRadius: '50%', marginRight: '6px' }}></span>
              <span style={{ fontWeight: '500' }}>LSTM: </span>
              <span style={{ marginLeft: '4px', color: 'blue' }}>{legendData.lstm}</span>
            </div>
          )}
          {legendData.gcn && visibilityState.gcn && (
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <span style={{ width: '10px', height: '10px', backgroundColor: 'brown', borderRadius: '50%', marginRight: '6px' }}></span>
              <span style={{ fontWeight: '500' }}>GCN: </span>
              <span style={{ marginLeft: '4px', color: 'brown' }}>{legendData.gcn}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CandlestickChart;
