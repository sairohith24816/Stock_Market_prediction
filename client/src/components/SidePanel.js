import React from "react";

const SidePanel = () => {
  return (
    <div className="bg-gray-50  ">
      <ul className="flex flex-col space-y-8  mx-8 p-4 overflow-hidden z-10">
        <li> FUTURES </li>
        <li> OPTIONS </li>
        <li> STRATEGY </li>
        <li> BACKTEST </li>
        <li> ABOUT </li>
      </ul>
    </div>
  );
};

export default SidePanel;
