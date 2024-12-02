const getTransactions = async () => {
    try {
      const response = await fetch('http://localhost:8080/getAllTransactions');
      const data = await response.json();
      console.log("Received data:", data);

      // Process raw transaction data
      const processedData = data.map(item => {
        const transactions = item.transactions.map(txHex => {
          try {
            // Convert hex to bytes
            const txBytes = new Uint8Array(
              txHex.match(/.{1,2}/g).map(byte => parseInt(byte, 16))
            );

            // Extract function name (first 18 bytes)
            let funcName = "";
            for (let i = 0; i < 18; i++) {
              if (txBytes[i] !== 0) {
                funcName += String.fromCharCode(txBytes[i]);
              }
            }

            // Extract source public key (next 33 bytes)
            const compressedPubKey = txBytes.slice(18, 51);
            const srcPk = Array.from(compressedPubKey)
              .map(byte => byte.toString(16).padStart(2, "0"))
              .join("");

            // Extract timestamp (next 8 bytes)
            let timestamp = 0n;
            for (let i = 0; i < 8; i++) {
              timestamp |= BigInt(txBytes[51 + i]) << BigInt(i * 8);
            }
            
            const timestampSec = Number(timestamp);
            if (!Number.isFinite(timestampSec)) {
              throw new Error('Invalid timestamp value');
            }
            
            // Convert seconds to milliseconds for JavaScript Date
            const timestampMs = timestampSec * 1000;

            return {
              raw_data: txHex,
              src_pk: srcPk,
              timestamp: timestampSec,
              timestampMs,
              func_name: funcName.trim()
            };
          } catch (error) {
            console.error(`Failed to process transaction: ${error.message}`);
            return null;
          }
        }).filter(tx => tx !== null);

        return {
          pk: item.pk,
          transactions
        };
      });

      // Generate pkData for BarChart
      const pkDataMap = new Map();
      processedData.forEach(item => {
        if (item.pk && item.transactions.length > 0) {
          const shortPk = item.pk.substring(0, 4);
          pkDataMap.set(shortPk, (pkDataMap.get(shortPk) || 0) + item.transactions.length);
        }
      });

      const pkData = Array.from(pkDataMap.entries())
        .map(([pk, count]) => ({
          pk: `PK${pk}`,
          count: Math.max(1, count)
        }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10);

      // Calculate TPS data with proper timestamp range
      const getAllTransactionDates = processedData.flatMap(item => 
        item.transactions.map(tx => new Date(tx.timestampMs))
      );

      const oldestDate = new Date(Math.min(...getAllTransactionDates));
      const newestDate = new Date(Math.max(...getAllTransactionDates));

      // Generate hourly time points between oldest and newest dates
      const timePoints = [];
      let currentDate = new Date(oldestDate);
      currentDate.setMinutes(0, 0, 0);

      while (currentDate <= newestDate) {
        timePoints.push(new Date(currentDate));
        currentDate.setHours(currentDate.getHours() + 1);
      }

      const tpsData = timePoints.map(hour => {
        const count = processedData.reduce((acc, item) => {
          return acc + item.transactions.filter(tx => {
            const txDate = new Date(tx.timestampMs);
            return txDate.getHours() === hour.getHours() &&
                  txDate.getDate() === hour.getDate() &&
                  txDate.getMonth() === hour.getMonth();
          }).length;
        }, 0);

        return {
          timestamp: hour.toISOString(),
          tps: count || 0
        };
      });

      // Generate time distribution
      const timeRanges = {
        '00-04': 0,
        '04-08': 0,
        '08-12': 0,
        '12-16': 0,
        '16-20': 0,
        '20-24': 0
      };

      processedData.forEach(item => {
        item.transactions.forEach(tx => {
          if (tx && tx.timestampMs) {
            const hour = new Date(tx.timestampMs).getHours();
            const rangeStart = Math.floor(hour / 4) * 4;
            const rangeKey = `${rangeStart.toString().padStart(2, '0')}-${(rangeStart + 4).toString().padStart(2, '0')}`;
            timeRanges[rangeKey]++;
          }
        });
      });

      const timeDistribution = Object.entries(timeRanges)
        .map(([name, value]) => ({
          name,
          value: Math.max(0, value)
        }));

      // Generate chatData
      const chatData = processedData.map(item => ({
        pk: item.pk,
        transactions: item.transactions.map(tx => ({
          raw_data: tx.raw_data,
          src_pk: tx.src_pk,
          timestamp: tx.timestamp.toString(),
          func_name: tx.func_name
        }))
      })

      );

      return {
        tpsData,
        pkData,
        timeDistribution,
        chatData
      };
    } catch (error) {
      console.error("Data fetch error:", error);
      throw error;
    }
  };

  export default getTransactions;