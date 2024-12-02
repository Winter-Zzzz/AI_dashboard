import React, { useState, useMemo } from 'react';
import { CHART_DIMENSIONS } from './chartDimensions';
import { Send } from 'lucide-react';

const ChatBox = ({ data }) => {
  const [messages, setMessages] = useState([]); 
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const { dataset } = useMemo(() => {
    if (!data?.chatData) {
      return { dataset: [] };
    }
    return { dataset: data.chatData };
  }, [data]);

  console.log("Received Data", dataset)

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
    messageResponse: {
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
    codeBlock: {
      fontFamily: 'monospace',
      fontSize: '12px',
      backgroundColor: '#2d2d2d',
      color: '#e6e6e6',
      padding: '12px',
      borderRadius: '4px',
      whiteSpace: 'pre-wrap',
      wordBreak: 'break-all'
    },
    resultBlock: {
      marginTop: '8px',
      fontFamily: 'monospace',
      fontSize: '12px',
      backgroundColor: '#1a1a1a',
      color: '#e6e6e6',
      padding: '12px',
      borderRadius: '4px',
      maxHeight: '300px',
      overflowY: 'auto',
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading) return;
    setIsLoading(true);
  
    const userMessage = {
      id: Date.now(),
      text: inputText, 
      isUser: true 
    };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInputText('');
  
    try {
      const queryText = encodeURIComponent(inputText);
      const encodedDataset = encodeURIComponent(JSON.stringify(dataset));
      const response = await fetch(`http://localhost:8000/api/query-transactions?query_text=${queryText}&dataset=${encodedDataset}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error('서버 오류가 발생했습니다.');
      }
  
      const responseData = await response.json();
      console.log('서버 응답:', responseData); // 디버깅용
  
      if (responseData.status === 'success' && responseData.data) {
        const responseMessage = {
          id: Date.now() + 1,
          text: `트랜잭션 조회 결과:`,
          result: JSON.stringify(responseData.data.transactions, null, 2),
          // code: responseData.data.generated_code,
          isUser: false
        };
        setMessages(prevMessages => [...prevMessages, responseMessage]);
      } else {
        const noDataMessage = {
          id: Date.now() + 1,
          text: '검색 결과가 없습니다. 다른 검색어로 시도해보세요.',
          isUser: false
        };
        setMessages(prevMessages => [...prevMessages, noDataMessage]);
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: `오류: ${error.message}`,
        isUser: false,
        isError: true
      };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
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
                ...(message.isUser ? styles.userMessage : styles.responseMessage),
                ...(message.isError && { backgroundColor: '#ff5555' })
              }}
            >
              <div>{message.text}</div>
              {message.code && (
                  <div style={styles.codeBlock}>{message.code}</div>
              )}
              {message.result && (
                <div style={styles.resultBlock}>
                  {message.result}
                </div>
              )}
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
          placeholder="자연어로 트랜잭션을 설명해주세요"
        />
        <button type="submit" style={styles.button} disabled={isLoading}>
          <Send size={20} />
        </button>
      </form>
    </div>
  );
};

export default ChatBox;