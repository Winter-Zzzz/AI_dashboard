import  React from 'react';
import { PieChart } from '@mui/x-charts/PieChart';
import { Typography, Box } from '@mui/material';
import { CHART_DIMENSIONS } from './chartDimensions';

const timeColors = {
  '00-04': '#26a69a',
  '04-08': '#42a5f5',
  '08-12': '#7e57c2',
  '12-16': '#5c6bc0',
  '16-20': '#ec407a',
  '20-24': '#66bb6a'
};

const PieChartComponent = ({ data }) => {
  // Process data for the pie chart
  const chartData = React.useMemo(() => {
    if (!data?.timeDistribution) {
      return [];
    }

    return data.timeDistribution.map(item => ({
      id: item.name,
      value: item.value,
      label: item.name,
      color: timeColors[item.name] || '#999999'
    }));
  }, [data]);

  const valueFormatter = (value) => `${value} TPS`;
  return (
    <Box sx={{ 
      width: CHART_DIMENSIONS.width, 
      height: CHART_DIMENSIONS.height, 
      position: 'relative', 
    }}>
      <Typography variant="h6" align="center" sx={{ mb: 2 }}>
        TPS Distribution by Time Period
      </Typography>
      <PieChart
        width={CHART_DIMENSIONS.width}
        height={CHART_DIMENSIONS.height-55}
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
            arcLabelRadius: 70,
            valueFormatter,
            outerRadius: 120
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

export default PieChartComponent;
