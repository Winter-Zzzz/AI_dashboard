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
            color: '#4DB6AC',
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

/*
export default function StackedLineChart() {
  return (
      <LineChart
        dataset={dataset}
        xAxis={[
          {
            id: 'Years',
            dataKey: 'date',
            scaleType: 'time',
            valueFormatter: (date) => date.getFullYear().toString(),
          },
        ]}
        series={[
          {
            id: 'France',
            label: 'French GDP per capita',
            dataKey: 'fr',
            stack: 'total',
            area: true,
            showMark: false,
            ccolor: '#4DB6AC'
          },
          {
            id: 'Germany',
            label: 'German GDP per capita',
            dataKey: 'dl',
            stack: 'total',
            area: true,
            showMark: false,
            color: '#64B5F6'
          },
          {
            id: 'United Kingdom',
            label: 'UK GDP per capita',
            dataKey: 'gb',
            stack: 'total',
            area: true,
            showMark: false,
            color: '#BA68C8'
          },
        ]}
        width={ 500 }
        height={ 400 }
        sx={{
          '.MuiLineElement-root': {
            display: 'none',
          },
          '.MuiChartsLegend-label': {
            fontSize: '0.8rem',
          },
          '.MuiChartsLegend-root': {
            transform: 'scale(0.85)',
            transformOrigin: 'center',
          }
        }}
    />
  );
}
*/
