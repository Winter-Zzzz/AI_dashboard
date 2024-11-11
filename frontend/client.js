import React from 'react';
import ReactDOM from 'react-dom';
import BarChartComponent from './components/BarChartComponent';
import PieChartComponent from './components/PieChartComponent';
import StackedLineChart from './components/StackedLineChart';

// 부모 컴포넌트 생성
const App = () => {
  return (
    <div>
      {/* 세로로 배치하는 flexbox */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <BarChartComponent />
        <PieChartComponent />
        <StackedLineChart />
      </div>
    </div>
  );
};

// 'root' 엘리먼트에 App 컴포넌트를 렌더링
ReactDOM.render(<App />, document.getElementById('root'));
