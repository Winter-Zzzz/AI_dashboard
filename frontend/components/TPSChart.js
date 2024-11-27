import React, { useMemo } from 'react';
import { LineChart } from '@mui/x-charts/LineChart';
import { CHART_DIMENSIONS } from './chartDimensions';

export default function TPSChart({ data }) {  
  const { dataset, maxTPS } = useMemo(() => {
    if(!data?.tpsData) {
      return { dataset: [], maxTPS: 400 };
    }

    const formattedData = data.tpsData.map(item => ({
      date: new Date(item.timestamp),
      tps: item.tps
    }));

    const currentMax = Math.max(...formattedData.map(item => item.tps));
    const calculatedMaxTPS = Math.ceil(currentMax / 100) * 100;

    return{
      dataset: formattedData,
      maxTPS: calculatedMaxTPS
    };
  }, [data]);

  return (
    <div className="w-full" style={{ minWidth: CHART_DIMENSIONS.width }}> 
      <LineChart
        dataset={dataset}
        margin={CHART_DIMENSIONS.margin}
        xAxis={[{
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
        }]}
        yAxis={[{
          id: 'TPS',
          scaleType: 'linear',
          valueFormatter: (value) => value.toFixed(0),
          min: 0,
          max: maxTPS
        }]}
        series={[
          {
            id: 'TPS',
            label: 'Transactions Per Second',
            dataKey: 'tps',
            curve: "linear",
            showMark: true,
            color: '#A50034',
            markerSize: 6
          }
        ]}
        slotProps={{
          legend: {
            position: {
              vertical: 'top',
              horizontal: 'middle'
            },
          }
        }}
        width={CHART_DIMENSIONS.width}
        height={CHART_DIMENSIONS.height}
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
      />
    </div>
  )
};