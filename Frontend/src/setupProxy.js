const { createProxyMiddleware } = require('http-proxy-middleware')

module.exports = function (app) {
  console.log('matching')
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'https://hz.matpool.com:27479',
      changeOrigin: true,
      pathRewrite: {
        '/api': ''
      }
    })
  )
}
