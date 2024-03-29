const express = require('express')
const app = express()
const port = 3000

app.use(express.static('web'))
app.use(function(req, res, next) {
    res.header('Access-Control-Allow-Origin', '*');
    res.header(
        'Access-Control-Allow-Headers',
        'Origin, X-Requested-With, Content-Type, Accept'
    );
    next();
});

app.listen(port, () => {
    console.log(`Example app listening at http://localhost:${port}`)
})