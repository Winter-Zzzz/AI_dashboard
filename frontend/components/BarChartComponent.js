import React from 'react';
import { BarChart } from '@mui/x-charts/BarChart';
import { dataset, valueFormatter } from '../dataset/transactions';
import { CHART_DIMENSIONS } from './chartDimensions';

export default function BarChartComponent() {

    const chartSetting = {
        xAxis: [
            {
                label: 'transactions',
            },
        ],
        width: CHART_DIMENSIONS.width,
        height: CHART_DIMENSIONS.height,
        margin: CHART_DIMENSIONS.margin
    };

    return ( <BarChart
        dataset = {dataset}
        yAxis={[{ scaleType: 'band', dataKey: 'pk', barGapRatio: 0.8}]}
        xAxis={[{ 
            label: 'Transaction Count',
            min: 0,
            max: 12,
            tickSize: 10,
          }]}
        series={[{dataKey: 'transactions', label: 'Transactions by PK', valueFormatter, color: '#748BA7' }]}
        layout="horizontal"
        {...chartSetting}
    />
);
}