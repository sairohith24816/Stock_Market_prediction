// getDate.js

function calculateStartDate(time_period) {
    const currentDate = new Date();
  
    switch (time_period) {
      case '1Day':
        return new Date(currentDate - 24 * 60 * 60 * 1000);
      case '1Week':
        return new Date(currentDate - 7 * 24 * 60 * 60 * 1000);
      case '1Month':
        return new Date(currentDate - 30 * 24 * 60 * 60 * 1000);
      default:
        throw new Error('Invalid time period');
    }
  }
  
  module.exports = { calculateStartDate };
  