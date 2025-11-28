import React from "react";

const TimeHeader = () => {
  return (
    <div className="flex bg-blue-300 text-blue-700 font-bold justify-between">
      <ul className="flex space-x-2  mx-2 ">
        <li>5 min</li>
        <li>15 min</li>
        <li>1 hr</li>
        <li>1 day</li>
        <li>1 day</li>
      </ul>

      {/* <ul className="flex space-x-2 mx-2">
        <li>Line Chart</li>
        <li>Candlesticks Chart</li>
      </ul> */}
    </div>
  );
};

export default TimeHeader;
