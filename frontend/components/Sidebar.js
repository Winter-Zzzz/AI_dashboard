import React from 'react';

const Sidebar = ({ activeMenu, onMenuClick } ) => {
    const menuItems = [
        { id: 'Dashboard', icon: '🏠', label: 'Dashboard' },
        { id: 'Products', icon: '📦', label: 'Products' },
        { id: 'Customers', icon: '👥', label: 'Customers' },
        { id: 'Orders', icon: '🛍️', label: 'Orders' },
        { id: 'Analytics', icon: '📊', label: 'Analytics' },
        { id: 'Marketing', icon: '📢', label: 'Marketing' },
        { id: 'Discounts', icon: '🏷️', label: 'Discounts' },
        { id: 'Payouts', icon: '💰', label: 'Payouts' },
        { id: 'Statements', icon: '📄', label: 'Statements' },
        { id: 'Calendar', icon: '📅', label: 'Calendar' },
        { id: 'Storefront', icon: '🏪', label: 'Storefront' }
    ]

    return (
        <div style={{
            width: '250px',
            height: '100%',
            backgroundColor: '#ffffff',
            borderRight: '1px solid black',
            padding: '60px 0',
            position: 'fixed',
            fontSize: '20px',
        }}>
            {menuItems.map((item) => (
                <div
                    key = {item.id}
                    onClick={() => onMenuClick(item.id)}
                    style = {{
                        padding: '15px 20px',
                        cursor: 'pointer',
                        backgroundColor: activeMenu === item.id ? '#2a2a2a' : 'transparent',
                        color: activeMenu === item.id ? 'white' : 'inherit',
                        display: 'flex',
                        alignItems: 'center', 
                        gap: '10px', 
                        transition: 'background-color 0.2s',
                    }}
                    onMouseEnter={(e) => {
                        if (activeMenu !== item.id) {
                            e.currentTarget.style.backgroundColor = '#f5f5f5';
                        }
                    }}
                    onMouseLeave={(e) => {
                        if (activeMenu !== item.id) {
                            e.currentTarget.style.backgroundColor = 'transparent';
                        }
                    }}
                >
                    <span>{item.icon}</span>
                    <span>{item.label}</span>
                </div>
            ))}
        </div>
    );
};



export default Sidebar;