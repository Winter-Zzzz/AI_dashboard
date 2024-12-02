import React, { useState, useMemo, useCallback } from 'react';
import { CHART_DIMENSIONS } from './chartDimensions';
import { Send } from 'lucide-react';

const ChatBox = ({ data }) => {
  const [messages, setMessages] = useState([]); 
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [messageCounter, setMessageCounter] = useState(0); // 메시지 카운터 추가


  const dataset = useMemo(() => {
    return data?.chatData || [];
  }, [data?.chatData]);

  const styles = useMemo(() => ({
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
      wordBreak: 'break-word', // Added to handle long text better

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
    },

    // ==== 여기서부터 추가함 
    displayTxn: {
      marginTop: '8px',
      fontSize: '14px',
      wordBreak: 'break-word',  // 트랜잭션 데이터 줄바꿈
    },
    infoRow: {
      display: 'flex',
      flexDirection: 'column',  // 세로로 배치하여 공간 효율성 향상
      marginBottom: '4px',
    },
    messageText: {
      maxWidth: '100%',
      display: 'block'
    },
}), [isLoading]);

  const truncateText = (text) => {
    if (!text || typeof text !== 'string') return '';
    const maxLength = 32;
    return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
  };

  const handleSendMessage = useCallback(async (e) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading) return;
    setIsLoading(true);

    // 사용자 메세지 
    setMessageCounter(prev => prev + 1)
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
        const resultMessages = responseData.data.transactions.flatMap((item, itemIndex) => {
          return item.transactions.map((tx, txIndex) => {
            setMessageCounter(prev => prev + 1);
            return {
              id: `tx-${itemIndex}-${txIndex}-${messageCounter}`, // 고유한 key 생성
              transactionData: {
                Time: new Date(parseInt(tx.timestamp) * 1000).toLocaleString(),
                pk: truncateText(item.pk, 20),
                src_pk: truncateText(tx.src_pk, 20),
                Function: tx.func_name,
              },
              isUser: false
            };
          });
        });

        console.log("resultMessages", resultMessages)
        setMessages(prevMessages => [...prevMessages, ...resultMessages]);
      } else {
        setMessages(prevMessages => [...prevMessages, {
          id: `nodata-${messageCounter}`,
          text: '검색 결과가 없습니다. 다른 검색어로 시도해보세요.',
          isUser: false
        }]);
      }
    } catch (error) {
      setMessages(prevMessages => [...prevMessages, {
        id: `error-${messageCounter}`,
        text: `오류: ${error.message}`,
        isUser: false,
        isError: true
      }]);
    } finally {
      setIsLoading(false);
    }
  }, [inputText, isLoading, dataset, messageCounter]);

  const handleInputChange = useCallback((e) => {
    setInputText(e.target.value);
  }, []);

  const DisplayTransaction = React.memo(({ data }) => (
    <div style={styles.displayTxn}>
    {Object.entries(data).map(([key, value]) => (
      <div key={key} style={styles.infoRow}>
        <span style={styles.label}>{key}:</span>
        <span>{value}</span>
      </div>
    ))}
  </div>
));

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
            {message.text && <div>{message.isUser ? truncateText(message.text, 100) : message.text}</div>}
            {message.transactionData && (
              <DisplayTransaction key={`txn-${message.id}`} data={message.transactionData} />
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
        onChange={handleInputChange}
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

export default React.memo(ChatBox)