const { app, BrowserWindow } = require('electron');
const path = require('path');

// electron-is-dev 대신 직접 개발 환경 체크
const isDev = process.env.NODE_ENV === 'development' || process.defaultApp;

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true,
    },
  });

  if (isDev) {
    win.loadURL('http://localhost:8080');
    win.webContents.openDevTools();
  } else {
    win.loadFile(path.join(__dirname, 'dist', 'index.html'));
  }

  if (isDev) {
    win.webContents.on('did-frame-finish-load', () => {
      win.webContents.on('before-input-event', (event, input) => {
        if (input.control && input.key.toLowerCase() === 'i') {
          win.webContents.toggleDevTools();
        }
      });
    });
  }
}

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});