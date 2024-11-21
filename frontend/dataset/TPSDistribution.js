export const tpsData = [
  { label: '00-04', value: 150 },
  { label: '04-08', value: 280 },
  { label: '08-12', value: 420 },
  { label: '12-16', value: 380 },
  { label: '16-20', value: 350 },
  { label: '20-24', value: 220 }
];

export const valueFormatter = (item) => `${item.value} TPS`;

export const getTPSColor = (label) => {
  const colors = {
    '00-04': '#26a69a',
    '04-08': '#42a5f5',
    '08-12': '#7e57c2',
    '12-16': '#5c6bc0',
    '16-20': '#ec407a',
    '20-24': '#66bb6a'
  };
  return colors[label] || '#8d6e63';
};