import React, { useState } from 'react';
import { CHART_DIMENSIONS } from './chartDimensions';
import { Send } from 'lucide-react';

const ChatBox = () => {
  const [messages, setMessages] = useState([]); 
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const styles = {
    container: {
      width: CHART_DIMENSIONS.width,
      height: CHART_DIMENSIONS.height,
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: '#f8f9fa',
      borderRadius: '8px',
      boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    },
    messageArea: {
      flex: 1,
      overflowY: 'auto',
      padding: '16px',
    },
    messageContainer: {
      marginBottom: '16px',
      display: 'flex',
    },
    messageUser: {
        justifyContent: 'flex-end',
    },
    messageRespond: {
        justifyContent: 'flex-start',
    },
    message: {
      maxWidth: '70%',
      padding: '12px',
      borderRadius: '8px',
    },
    userMessage: {
      backgroundColor: '#748BA7',
      color: 'white',
    },
    responseMessage: {
        backgroundColor: '#E7E8E8',
        color: '#748BA7'
    },
    inputArea: {
      borderTop: '1px solid #eee',
      padding: '16px',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
    },
    input: {
      flex: 1,
      padding: '8px',
      border: '1px solid #ddd',
      borderRadius: '8px',
      outline: 'none',
      fontSize: '14px',
    },
    button: {
      padding: '8px',
      backgroundColor: '#748BA7',
      border: 'none',
      borderRadius: '8px',
      color: 'white',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      opacity: isLoading ? 0.7 : 1,
    },
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading) return;

    setIsLoading(true);

    // Add user message
    const userMessage = {
        id: Date.now(),
        text: inputText, 
        isUser: true 
    };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInputText('');

    setTimeout(() => {
      const responseMessage = {
        id: Date.now() + 1,
        text: 'transactions: ',
        isUser: false
      };
      setMessages(prevMessages => [...prevMessages, responseMessage]);
      setIsLoading(false);
    }, 1000);
  };

  return (
    <div style={styles.container}>
      <div style={styles.messageArea}>
        {messages.map(message => (
          <div
            key={message.id}
            style={{
              ...styles.messageContainer,
              ...(message.isUser ? styles.messageUser : styles.messageResponse)
            }}
          >
            <div
              style={{
                ...styles.message,
                ...(message.isUser ? styles.userMessage : styles.responseMessage)
              }}
            >
              {message.text}
            </div>
          </div>
        ))}
      </div>
      <form style={styles.inputArea} onSubmit={handleSendMessage}>
        <input
          type="text"
          style={styles.input}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          disabled={isLoading}
        />
        <button type="submit" style={styles.button} disabled={isLoading}>
          <Send size={20} />
        </button>
      </form>
    </div>
  );
};

export default ChatBox;