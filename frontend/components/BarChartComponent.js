import React from 'react';
import { BarChart } from '@mui/x-charts/BarChart';
import { CHART_DIMENSIONS } from './chartDimensions';

const BarChartComponent = ({ data }) => {
    const dataset = React.useMemo(()=> {
        if(!data?.pkData) {
            return[];
        }
    return data.pkData.map(item => ({
        pk: item.pk.substring(0, 10),
        transactions: item.count
        }));
    }, [data]);

    const valueFormatter = (value) => `${value}`

    const customMargin = {
        ...CHART_DIMENSIONS.margin,
        left: 50  // 왼쪽 여백 줄임
      };
    
    const chartSetting = {
        xAxis: [
            {
                label: 'transactions',
                min: 0,
                max: 8, 
            },
        ],
        width: CHART_DIMENSIONS.width,
        height: CHART_DIMENSIONS.height,
        margin: customMargin
    };


    return ( 
        <BarChart
            dataset = {dataset}
            yAxis={[{ scaleType: 'band', dataKey: 'pk', barGapRatio: 0.8, tickLabelStyle: { fontSize: '0.7rem' }}]}
            xAxis={[{ 
                label: 'Transaction Count',
            }]}
            series={[{dataKey: 'transactions', label: 'Transactions by PK', valueFormatter, color: '#748BA7' }]}
            layout="horizontal"
            {...chartSetting}
            sx={{
                '& .MuiChartsAxis-tick': {
                  display: 'block',
                  maxWidth: '70px'  // tick 컨테이너의 최대 너비 설정
                },
            }}
        />
    );
}

export default BarChartComponent;