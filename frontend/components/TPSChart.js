import React from 'react';
import { LineChart } from '@mui/x-charts/LineChart';
import { dataset } from '../dataset/TPS';
import { CHART_DIMENSIONS } from './chartDimensions';

export default function TPSChart() {  
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
            const hours = String(date.getHours()).padStart(2, '0');
            return `${hours}:00:00`;
          },
          tickNumber: 7
        }]}
        yAxis={[{
          id: 'TPS',
          scaleType: 'linear',
          valueFormatter: (value) => value.toFixed(0),
          min: 0,
          max: 400
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
            fontsize: '0.8rem',
          },
          '.MuiChartsLegend-root': {
            transform: 'scale(0.85)',
            tranformOrigin: 'center',
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