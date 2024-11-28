import React, { useMemo } from 'react';
import { Users, ArrowRightLeft } from 'lucide-react';
import { CHART_DIMENSIONS } from './chartDimensions';

const MetricCards = ({ data }) => {
  const metrics = useMemo(() => {
    const totalTransactions = data.pkData?.reduce((sum, item) => sum + item.count, 0) || 0;
    const totalUsers = data.pkData?.length || 0;
    return { totalTransactions, totalUsers };
  }, [data]);

  // Calculate actual content dimensions after margins
  const contentWidth = CHART_DIMENSIONS.width - CHART_DIMENSIONS.margin.left - CHART_DIMENSIONS.margin.right;
  const contentHeight = CHART_DIMENSIONS.height - CHART_DIMENSIONS.margin.top - CHART_DIMENSIONS.margin.bottom;

  const containerStyle = {
    width: CHART_DIMENSIONS.width,
    height: CHART_DIMENSIONS.height,
    margin: CHART_DIMENSIONS.margin,
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center'
  };

  const cardContainerStyle = {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    width: contentWidth,
    height: contentHeight,
    justifyContent: 'center'
  };

  const cardStyle = {
    backgroundColor: '#f8f9fa',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    padding: '24px',
    flex: '0 1 auto'
  };

  const cardContentStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  };

  const labelStyle = {
    color: '#666',
    fontSize: '20px',
    marginBottom: '8px'
  };

  const valueStyle = {
    fontSize: '32px',
    fontWeight: 600
  };

  const iconContainerStyle = {
    padding: '12px',
    borderRadius: '8px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  };

  return (
    <div style={containerStyle}>
      <div style={cardContainerStyle}>
        <div style={cardStyle}>
          <div style={cardContentStyle}>
            <div>
              <div style={labelStyle}>Total Transactions</div>
              <div style={valueStyle}>
                {metrics.totalTransactions.toLocaleString()}
              </div>
            </div>
            <div style={{
              ...iconContainerStyle,
              backgroundColor: 'rgba(59, 130, 246, 0.1)'
            }}>
              <ArrowRightLeft size={24} color="#2563eb" />
            </div>
          </div>
        </div>
        <div style={cardStyle}>
          <div style={cardContentStyle}>
            <div>
              <div style={labelStyle}>Total Users</div>
              <div style={valueStyle}>
                {metrics.totalUsers.toLocaleString()}
              </div>
            </div>
            <div style={{
              ...iconContainerStyle,
              backgroundColor: 'rgba(168, 85, 247, 0.1)'
            }}>
              <Users size={24} color="#7c3aed" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MetricCards;