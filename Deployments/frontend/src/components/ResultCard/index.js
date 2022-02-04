/* Importing packages */
import React from 'react'
import { Card } from '@mui/material';
import { CardContent }from '@mui/material';
import { Typography } from '@mui/material';
import { Divider } from '@mui/material';
import { CardHeader } from '@mui/material';
import { Box } from '@mui/material';
import { makeStyles } from '@material-ui/core';

/* ResultCard component captures result for the top prediciton */
const ResultCard = ({mainResult}) => {
    
    /* Styling content */
    const useStyles = makeStyles(theme => ({
    root: {
        borderRadius: 12,
        minWidth: 256,
        textAlign: 'center',
        backgroundColor: '#eceff1',
    },
    header: {
        textAlign: 'center',
        spacing: 10,
    },
    list: {
        padding: '20px',
    },
    action: {
        display: 'flex',
        justifyContent: 'space-around',
    },
    }));
    

    const classes = useStyles();

    /* Determining what colour to highligh the text with, breakpoints at each increment of 33.33%  */
    const colorCode = value => {

        const parsedValue = parseFloat(value)

        if (parsedValue < 33.33) {
            return '#ec1a1a'
        } else

        if (parsedValue < 66.66) {
            return '#ffb003'
        } 

        return '#01bf71'

    }
    
    return (
       
            <Card elevation={5} className={classes.root}>
            <CardHeader title = {<strong>Top Prediction</strong>} className={classes.header} />
            <Divider variant="middle" />
            <CardContent>
            <Typography noWrap variant="h5" gutterBottom align="left">
                <strong>SSOC Title:</strong> {mainResult.top_prediction ? (mainResult.top_prediction.SSOC_Title) : 'xxx'}
            </Typography>
            <Typography noWrap variant="h5" gutterBottom align="left">
                <strong>SSOC Code:</strong> {mainResult.top_prediction ? (mainResult.top_prediction.SSOC_Code) : 'xxx'}
            </Typography>
            <Typography noWrap variant="h5" gutterBottom align="left" color = {colorCode(mainResult.top_prediction.Prediction_Confidence)}>
                <strong>Confidence:</strong> {mainResult.top_prediction ? (mainResult.top_prediction.Prediction_Confidence) : 'xxx'}
            </Typography>   
            <Divider variant="middle"  sx={{ my: 2 }}/>
            <Typography noWrap variant="h5" gutterBottom align="center">
                <strong>SSOC Description:</strong> 
            </Typography>
            <Box component="div" sx={{ maxHeight: '850px', overflow: 'auto', m: '1rem' }}>
            <Typography align="left">{mainResult.top_prediction ? (mainResult.top_prediction.SSOC_Description) : 'xxx'}</Typography>
            </Box>
            </CardContent>
            </Card>
        
    )
}

export default ResultCard
