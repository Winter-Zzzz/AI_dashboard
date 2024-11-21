const HtmlWebpackPlugin = require('html-webpack-plugin');
const path = require('path');

module.exports = {
  entry: './App.js',  // 엔트리 파일 설정
  output: {
    path: path.resolve(__dirname, 'dist'),  // 출력 경로
    filename: 'app.js',  // 출력 파일 이름
  },
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,  // .js와 .jsx 파일을 처리
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',  // Babel 로더 사용
          options: {
            presets: [
              '@babel/preset-env',  // 최신 JavaScript 문법 변환
              '@babel/preset-react'  // React JSX 변환
            ]
          }
        }
      },
      {
        test: /\.css$/,  // .css 파일 처리
        use: ['style-loader', 'css-loader']
      }
    ]
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: './index.html'  // HTML 템플릿 설정
    })
  ],
  devServer: {
    port: 8080,  // 개발 서버 포트
    hot: true,  // Hot Module Replacement 활성화
    open: false  // 서버 시작 시 자동으로 브라우저 열지 않기
  },
  resolve: {
    extensions: ['.js', '.jsx'],  // .js, .jsx 확장자를 생략 가능
    fallback: {
      path: require.resolve('path-browserify'),
      stream: require.resolve('stream-browserify'),
      util: require.resolve('util/'),
      buffer: require.resolve('buffer/'),
      process: require.resolve('process/browser')
    }
  }
};
