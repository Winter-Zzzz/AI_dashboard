import React, { useState } from 'react';
import {
    Home, Package, Users, ShoppingCart, BarChart2, Megaphone,
    Tag, Wallet, FileText, Calendar, Store
} from 'lucide-react';

const styles = {
    sidebar: {
        width: '256px',
        height: '100%',
        backgroundColor: 'white',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
    },
    menuItem: {
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        padding: '30px 24px',
        color: '#4B5563',
        cursor: 'pointer',
        transition: 'all 0.2s ease'
    },
    menuItemActive: {
        backgroundColor: '#F9FAFB',
        color: '#A50034'
    },
    icon: {
        width: '20px',
        height: '20px'
    },
    label: {
        marginLeft: '16px',
        fontSize: '16px',
        fontWeight: '500'
    },
    highlight: {
        position: 'absolute',
        left: 0,
        width: '4px',
        height: '32px',
        backgroundColor: '#A50034',
        borderTopRightRadius: '4px',
        borderBottomRightRadius: '4px',
        opacity: 0,
        transition: 'opacity 0.2s ease'
    },
    highlightActive: {
        opacity: 1
    }
};

const Sidebar = () => {
    const [activeItem, setActiveItem] = useState(null);

    const menuItems = [
        { id: 'dashboard', icon: Home, label: 'Dashboard' },
        { id: 'products', icon: Package, label: 'Products' },
        { id: 'customers', icon: Users, label: 'Customers' },
        { id: 'orders', icon: ShoppingCart, label: 'Orders' },
        { id: 'analytics', icon: BarChart2, label: 'Analytics' },
        { id: 'marketing', icon: Megaphone, label: 'Marketing' },
        { id: 'discounts', icon: Tag, label: 'Discounts' },
        { id: 'payouts', icon: Wallet, label: 'Payouts' },
        { id: 'statements', icon: FileText, label: 'Statements' },
        { id: 'calendar', icon: Calendar, label: 'Calendar' },
        { id: 'storefront', icon: Store, label: 'Storefront' }
    ];

    const handleMouseEnter = (e) => {
        if (e.currentTarget.id !== activeItem) {
            e.currentTarget.style.backgroundColor = 'F9FAFB';
            e.currentTarget.style.color = "#A50034";
            e.currentTarget.querySelector('.highlight').style.opacity = '1';
        }
    };

    const handleMouseLeave = (e) => {
        if (e.currentTarget.id !== activeItem) {
            e.currentTarget.style.backgroundColor = 'transparent';
            e.currentTarget.style.color = '#4B5563';
            e.currentTarget.querySelector('.highlight').style.opacity = '0';
        }
    };

    const handleClick = (id) => {
        setActiveItem(id);
    };

    return (
        <div style={styles.sidebar}>
            <nav style={styles.nav}>
                {menuItems.map((item, index) => {
                    const isActive = activeItem === item.id;
                    const itemStyle = {
                        ...styles.menuItem,
                        ...(isActive && styles.menuItemActive)
                    };

                    return (
                        <div
                            key={item.id}
                            id={item.id}
                            style={itemStyle}
                            onMouseEnter={handleMouseEnter}
                            onMouseLeave={handleMouseLeave}
                            onClick={() => handleClick(item.id)}
                        >
                            <div 
                                className="highlight" 
                                style={{
                                   ...styles.highlight,
                                   ...(isActive && styles.highlightActive)
                                }}
                            />
                        <item.icon style={styles.icon} />
                        <span style={styles.label}>{item.label}</span>
                    </div>
                    );
                })}
            </nav>
        </div>
    );
};

export default Sidebar;