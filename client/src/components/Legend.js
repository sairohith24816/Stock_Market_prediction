import React from 'react';

const Legend = ({ 
  chartType = 'candlestick', 
  visibilityState = {}, 
  onToggleVisibility = () => {},
  isInHeader = false
}) => {
  const basePredictionItems = [
    { key: 'arima', color: 'orange', label: 'ARIMA Predictions', dotSize: '12px' },
    { key: 'lstm', color: 'blue', label: 'LSTM Predictions', dotSize: '12px' },
    { key: 'gcn', color: 'brown', label: 'GCN Predictions', dotSize: '12px' }
  ];

  // Use the same legend items for both chart types (no stock price toggle)
  const legendItems = basePredictionItems;

  const handleToggle = (key) => {
    onToggleVisibility(key, !visibilityState[key]);
  };

  const containerStyle = isInHeader ? {
    display: 'flex',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    border: '1px solid #ccc',
    borderRadius: '6px',
    padding: '8px 12px',
    fontSize: '12px',
    fontFamily: 'Arial, sans-serif',
    boxShadow: '0 2px 6px rgba(0, 0, 0, 0.15)',
    minWidth: 'auto'
  } : {
    position: 'absolute',
    top: '10px',
    right: '10px',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    border: '1px solid #ccc',
    borderRadius: '6px',
    padding: '12px',
    fontSize: '12px',
    fontFamily: 'Arial, sans-serif',
    boxShadow: '0 2px 6px rgba(0, 0, 0, 0.15)',
    zIndex: 1000,
    minWidth: '200px'
  };

  const titleStyle = isInHeader ? {
    fontWeight: 'bold',
    marginRight: '12px',
    fontSize: '13px'
  } : {
    fontWeight: 'bold',
    marginBottom: '8px',
    fontSize: '13px'
  };

  const itemStyle = isInHeader ? {
    display: 'flex',
    alignItems: 'center',
    marginRight: '12px',
    cursor: 'pointer',
    padding: '2px 4px',
    borderRadius: '3px',
    transition: 'background-color 0.2s'
  } : {
    display: 'flex',
    alignItems: 'center',
    marginBottom: '6px',
    cursor: 'pointer',
    padding: '2px',
    borderRadius: '3px',
    transition: 'background-color 0.2s'
  };

  return (
    <div style={containerStyle}>
      <div style={titleStyle}>
        Prediction Models
      </div>
      {legendItems.map((item, index) => (
        <div 
          key={index}
          style={{
            ...itemStyle,
            marginBottom: isInHeader ? '0' : (index === legendItems.length - 1 ? '0' : '6px'),
            marginRight: isInHeader ? (index === legendItems.length - 1 ? '0' : '12px') : '0'
          }}
          onClick={() => handleToggle(item.key)}
          onMouseEnter={(e) => e.target.style.backgroundColor = 'rgba(0, 0, 0, 0.05)'}
          onMouseLeave={(e) => e.target.style.backgroundColor = 'transparent'}
        >
          <input
            type="checkbox"
            checked={visibilityState[item.key] !== false}
            onChange={() => handleToggle(item.key)}
            style={{
              marginRight: '6px',
              cursor: 'pointer',
              transform: 'scale(0.9)'
            }}
            onClick={(e) => e.stopPropagation()}
          />
          <div
            style={{
              width: item.dotSize,
              height: item.dotSize,
              backgroundColor: visibilityState[item.key] !== false ? item.color : '#ccc',
              borderRadius: '50%',
              marginRight: '8px',
              flexShrink: 0,
              transition: 'background-color 0.2s'
            }}
          ></div>
          <span style={{ 
            color: visibilityState[item.key] !== false ? '#333' : '#999', 
            fontSize: '11px',
            transition: 'color 0.2s'
          }}>
            {item.label}
          </span>
        </div>
      ))}
    </div>
  );
};

export default Legend;
