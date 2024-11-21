import React from 'react';
import { CHART_DIMENSIONS } from './chartDimensions';

const TextBox = () => {
    return (
        <div style={{ 
            width: CHART_DIMENSIONS.width,
            height: CHART_DIMENSIONS.width,
            minWidth: CHART_DIMENSIONS.width
        }}>
            <textarea
                placeholder="텍스트를 입력하세요."
                style={{
                    width: '100%',
                    height: '100%',
                    padding: '15px',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    resize: 'none',
                    fontSize: '14px',
                    lineHeight: '1.5',
                    fontFamily: 'inherit',
                    boxSizing: 'border-box',
                    outline: 'none',
                    transition: 'border-color 0.2s ease',
                    '&:focus': {
                        borderColor: '#2196f3'
                    }
                }}
            />
        </div>
    );
};

export default TextBox;