import React from "react";
import SearchAndChartHeader from "./components/SearchAndChartHeader";


const App = () => {
  //change this to an api call and fetch data from there
  
  return (
    <div className="">
      {/* <h1 className="text-center text-3xl py-2 font-semibold my-2">
        TradingView
      </h1> */}

      {/* <h1 className="mx-8 font-">Candlestick Chart</h1> */}

      {/* <div className="container flex ">
        <input
          class=" mx-4  my-4 p-2 border border-gray-300 rounded-md  flex items-center justify-center"
          placeholder="Enter your stock"
        />
        <button className="bg-blue-700 text-white p-2 my-4 border-r-2 ">
          {" "}
          Search
        </button>
      </div> */}

    {/* <h1 className="flex justify-center text-4xl my-2 text-blue-600 ">CandleStick Chart</h1> */}
      {/* <TimeHeader></TimeHeader> */}
      <SearchAndChartHeader></SearchAndChartHeader>
     {/* <CandlestickChart selectedName={selname} /> */}
    </div>
  );
};

export default App;
