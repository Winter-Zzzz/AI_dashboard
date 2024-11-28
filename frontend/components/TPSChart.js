import React, { useMemo } from 'react';
import { LineChart } from '@mui/x-charts/LineChart';
import { CHART_DIMENSIONS } from './chartDimensions';

export default function TPSChart({ data }) {
  const { dataset } = useMemo(() => {
    if (!data?.tpsData) {
      return { dataset: [] };
    }

    const formattedData = data.tpsData.map((item, index) => ({
      date: new Date(item.timestamp),
      tps: item.tps,
      showMarker: index % 3 === 0
    }));

    return { dataset: formattedData };
  }, [data]);

  const chartSetting = {
    xAxis: [{
      id: 'Time',
      dataKey: 'date',
      scaleType: 'time',
      valueFormatter: (date) => {
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        const hours = String(date.getHours()).padStart(2, '0');
        return `${month}/${day} ${hours}:00`;
      },
      tickNumber: 7
    }],
    yAxis: [{
      id: 'TPS',
      scaleType: 'linear',
      valueFormatter: (value) => value.toFixed(0),
      min: 0,
      max: 8,
      tickNumber: 4  // 0, 2, 4, 6, 8을 표시하기 위한 설정
    }],
    width: CHART_DIMENSIONS.width,
    height: CHART_DIMENSIONS.height,
    margin: CHART_DIMENSIONS.margin,
    series: [{
      id: 'TPS',
      label: 'Transactions Per Second',
      dataKey: 'tps',
      curve: "linear",
      color: '#A50034',
      lineWidth: 1.5,
      marker: {
        size: 1,
        formatter: (params) => {
          return params.data.showMarker ? { style: { display: 'block' } } : { style: { display: 'none' } };
        }
      }
    }]
  };

  return (
    <div className="w-full" style={{ minWidth: CHART_DIMENSIONS.width }}>
      <LineChart
        dataset={dataset}
        slotProps={{
          legend: {
            position: {
              vertical: 'top',
              horizontal: 'middle'
            },
          }
        }}
        sx={{
          '.MuiChartsLegend-label': {
            fontSize: '0.8rem',
          },
          '.MuiChartsLegend-root': {
            transform: 'scale(0.85)',
            transformOrigin: 'center',
          },
          '& .MuiChartsAxis-label': {
            fontSize: '0.8rem',
          },
          '& .MuiChartsAxis-tickLabel': {
            fontSize: '0.75rem',
          }
        }}
        {...chartSetting}
      />
    </div>
  );
}