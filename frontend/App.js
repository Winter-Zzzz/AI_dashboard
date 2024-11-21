import React, { useState } from 'react';
import ReactDOM from 'react-dom/client';
import Sidebar from './components/Sidebar';
import TextBox from './components/TextBox';
import BarChartComponent from './components/BarChartComponent';
import PieChartComponent from './components/PieChartComponent';
import TPSChart from './components/TPSChart';

const App = () => {
  const [activeMenu, setActiveMenu] = useState('Dashboard');

  const handleMenuClick = (menuName) => {
    setActiveMenu(menuName);
  };

  return (
    <div style= {{ display: 'flex', height: '100%', overflow: 'hidden', backgroundColor: '#f8f9fa' }}>
      <div style={{ width: '250px', height: '100%', backgroundColor: 'white', boxshadow: '2px 0 5px rgba(0,0,0,0.1)' }}>
        <Sidebar activeMenu={activeMenu} onMenuClick={handleMenuClick} />
      </div>

      <div style={{ marginLeft: '250px', width: 'calc(100% - 250px)', padding: '20px', paddingBottom: 0 }}>
        <header style={{
          backgroundColor: 'transparent',
          color: 'black',
          padding: '10px 0',
          fontSize: '30px',
          fontweight: 'bold',
          marginBottom: '20px',
          whiteSpace: 'nowrap'
        }}>
          Matter Tunnel Dashboard
        </header>

        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '20px',
          marginBottom: 0
        }}>
          <div style={{
            display: 'flex',
            gap: '20px',
            width: '100%'
          }}>
            <div style={{
              flex: 1,
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              padding: '20px'
            }}>
              <TPSChart />
            </div>

            <div style={{
              flex: 1,
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              padding: '20px'
            }}>
              <TextBox />
            </div>
          </div>
          <div style={{
            display: 'flex',
            gap: '20px',
            width: '100%',
            marginBottom: 0
          }}>
            <div style={{
              flex: 1,
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              padding: '20px'
            }}>
              <BarChartComponent />
            </div>

            <div style={{
              flex: 1,
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              padding: '20px'
            }}>
              <PieChartComponent />
            </div>
          </div>
        </div>
      </div>
    </div>
  ); 
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);