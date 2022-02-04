import React from 'react'
import {
    ResultContainer,  
} from './ResultsElements';
import Grid from '@mui/material/Grid';
import ResultCard from '../ResultCard'
import AccordianResult from '../AccordianResult'


const ResultsSection = ({useStyles, mainResult}) => {


    return (
        <>
            <ResultContainer>
            <Grid container spacing={5} alignItems="center" justifyContent="center">
                
                <Grid item xs={1}>
                </Grid>

                <Grid item xs={5}>
                    <ResultCard useStyles = {useStyles} mainResult = {mainResult} originalText = "true"></ResultCard>
                </Grid>

                
                <Grid item xs={5}>
                    <ResultCard useStyles = {useStyles} mainResult = {mainResult} ></ResultCard>
                </Grid>

                <Grid item xs={1}>
                </Grid>
                
                <Grid item xs={1}>
                </Grid>

                <Grid item xs={10}>

                    <Grid container spacing={2} alignItems="center" justifyContent="center">
                    
                    {mainResult.next_9_preds?.map((nextpred, i) => ( // Mapping out all the predictions onto accordians      
                    <Grid item xs={4}>,
                    <AccordianResult useStyles = {useStyles} results = {nextpred} prob = {mainResult.next_9_proba[i]}/>
                    </Grid>
                    ))}
                    
                    </Grid>
                
                </Grid>

                <Grid item xs={1}>
                </Grid>
            </Grid>
            </ResultContainer>   

            
        </>
    )
}

export default ResultsSection
