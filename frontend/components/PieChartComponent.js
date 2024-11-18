import React from 'react';
import Box from '@mui/material/Box';
import { PieChart } from '@mui/x-charts/PieChart';
import { mobileAndDesktopOS, valueFormatter } from '../dataset/webUsageStats';

export default function PieChartComponent() {
  return (
    <Box sx={{ width: '70%' }}>
      <PieChart
        height={300}
        series={[
          {
            data: mobileAndDesktopOS, // 모든 데이터 사용
            innerRadius: 0, // 기본값으로 설정
            arcLabel: (params) => params.label ?? '', // 레이블 설정
            arcLabelMinAngle: 20,
            valueFormatter,
          },
        ]}
      />
    </Box>
  );
}
