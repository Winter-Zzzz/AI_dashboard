import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import { RotateCw } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChatBox from './components/ChatBox';
import BarChartComponent from './components/BarChartComponent';
import TPSChart from './components/TPSChart';
import getTransactions from './dataset/getTransactions';
import MetricCards from './components/MetricCards';

const App = () => {
  const [activeMenu, setActiveMenu] = useState('Dashboard');
  const [error, setError] = useState(null);
  const [transactionData, setTransactionData] = useState({
    tpsData: [],
    pkData: [],
    timeDistribution: [],
    chatData:[]
  });
  const [isLoading, setIsLoading] = useState(false);
  const [resetKey, setResetKey] = useState(0);

    const fetchData = async () => {
      setIsLoading(true)
      setResetKey(prev => prev + 1);
      try {
        const data = await getTransactions();
        console.log("Processed data:", data);
        setTransactionData(data);
        setError(null);
      } catch (error) {
        console.error("Data fetch error:", error);
        setError("Failed to load dashboard data");
        setTransactionData({
          tpsData: [],
          pkData: [],
          timeDistribution: [],
          chatData: []
        });
      } finally {
        setIsLoading(false);
      }
    };

    useEffect(() => {
      fetchData();
    }, []);


  const handleMenuClick = (menuName) => {
    setActiveMenu(menuName);
  };

  return (
    <div style= {{ display: 'flex', height: '100vh', overflow: 'hidden', backgroundColor: '#f8f9fa' }}>
      <div style={{ width: '250px', height: '100vh', backgroundColor: 'white', boxshadow: '2px 0 5px rgba(0,0,0,0.1)' }}>
        <Sidebar activeMenu={activeMenu} onMenuClick={handleMenuClick} />
      </div>

      <div style={{ marginLeft: '40px', width: 'calc(100% - 250px)', padding: '20px', paddingBottom: 0 }}>
      <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '15px',
          marginBottom: '20px'
        }}>
        <header style={{
          backgroundColor: 'transparent',
          color: 'black',
          padding: '10px 0',
          fontSize: '30px',
          fontWeight: 'bold',
          marginBottom: '0px',
          whiteSpace: 'nowrap',
        }}>
          Matter Tunnel Dashboard
        </header>

        <button
          onClick={fetchData}
          disabled={isLoading}
          style={{
            width: '40px',
            height: '40px',
            backgroundColor: 'transparent',
            border: '1px solid #A50034',
            borderRadius: '8px',
            cursor: isLoading ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 0,
            transition: 'all 0.2s ease',
            opacity: isLoading ? 0.5 : 1,
          }}
        >
          <RotateCw
            size={20}
            color='#A50034'
            className={isLoading ? 'animate-spin' : ''}
            style={{
              animation: isLoading ? 'spin 1s linear infinite' : 'none'
            }}
          />
        </button>
      </div>

        {error && (
          <div style={{
            color: 'red',
            padding: '10px',
            marginBottom: '10px',
            backgroundColor: '#fee',
            borderRadius: '4px'
          }}>
            {error}
          </div>
        )}

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
              flex: 'none',
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              padding: '20px',
              height: '350px',
              justifyContent: 'center',
              alignItems: 'center'
            }}>
              <TPSChart data={transactionData} />
            </div>

            <div style={{
              flex: 'none',
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              padding: '20px',
              height: '350px',
              justifyContent: 'center',
              alignItems: 'center'
            }}>
              <ChatBox key={resetKey} data={transactionData} />
            </div>
          </div>
          <div style={{
            display: 'flex',
            gap: '20px',
            width: '100%',
            marginBottom: 0
          }}>
            <div style={{
              flex: 'none',
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              padding: '20px',
              height: '350px',
              justifyContent: 'center',
              alignItems: 'center'
            }}>
              <BarChartComponent data={transactionData} />
            </div>

            <div style={{
              flex: 'none',
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              padding: '20px',
              height: '350px',
              justifyContent: 'center',
              alignItems: 'center'
            }}>
              <MetricCards data={transactionData} />
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