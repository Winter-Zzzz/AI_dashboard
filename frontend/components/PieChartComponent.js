import  React from 'react';
import { PieChart } from '@mui/x-charts/PieChart';
import { Typography, Box } from '@mui/material';
import { tpsData, getTPSColor, valueFormatter } from '../dataset/TPSDistribution';
import { CHART_DIMENSIONS } from './chartDimensions';

const chartData = tpsData.map(item => ({
  ...item,
  color: getTPSColor(item.label)
}))

export default function PieChartComponent() {
  return (
    <Box sx={{ 
      width: CHART_DIMENSIONS.width, 
      height: CHART_DIMENSIONS.height+ 50, 
      position: 'reltaive' 
    }}>
      <Typography variant="h6" align="center" sx={{ mb: 2 }}>
        TPS Distribution by Time Period
      </Typography>
      <PieChart
        width={CHART_DIMENSIONS.width}
        height={CHART_DIMENSIONS.height}
        series={[
          {
            data: chartData,
            innerRadius: 0,
            paddingAngle: 2,
            cornerRadius: 4, 
            startAngle: -90,
            highlightScope: { faded: 'global', highlighted: 'item' },
            faded: { innerRadius: 0, additionalRadius: -30, color: 'gray' },
            arcLabel: (item) => `${item.label}`,
            arcLabelMinAngle: 20,
            valueFormatter,
            outerRadius: 160
          },
        ]}
        legend={{
          direction: 'column',
          position: { vertical: 'middle', horizontal: 'right' },
          padding: 0,
          itemMarkWidth: 15,
          itemMarkHeight: 15,
          markGap: 5,
          itemGap: 10,
          labelStyle: {
            fontSize: 12,
          },
        }} 
        margin={CHART_DIMENSIONS.margin}
        slotProps={{
          legend: {
            itemMarkWidth: 16,
            itemMarkHeight: 16,
            markGap: 8,
            itemGap: 12,
          },
        }} 
      />
    </Box>
  );
}
