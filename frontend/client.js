import React from 'react';
import ReactDOM from 'react-dom';
import BarChartComponent from './components/BarChartComponent';
import PieChartComponent from './components/PieChartComponent';
import StackedLineChart from './components/StackedLineChart';

// 부모 컴포넌트 생성
const App = () => {
  return (
    <div style={{ width: '100%', height: '100vh', backgroundColor: '#f8f9fa' }}>
      {/* 상단 가로바 */}
      <header style={{
        backgroundColor: 'black', 
        color: 'white', 
        padding: '10px 20px', 
        fontSize: '20px', 
        fontWeight: 'bold',
        textAlign: 'center', 
      }}>
        Matter Tunnel Dashboard
      </header>

      {/* 그래프 배치 */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', width: '100%' }}>
          <BarChartComponent />
          <PieChartComponent />
        </div>

        <div style={{ marginTop: '20px', width: '100%', display: 'flex', justifyContent: 'center' }}>
          <StackedLineChart />
        </div>
      </div>
    </div>
  );
};

// 'root' 엘리먼트에 App 컴포넌트를 렌더링
ReactDOM.render(<App />, document.getElementById('root'));
