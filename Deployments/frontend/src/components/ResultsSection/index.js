/* Importing packages */
import React from 'react'
import { Box } from '@mui/material';
import { Grid } from '@mui/material';
import { Container } from '@mui/material';
import { Stack } from '@mui/material';
import { Divider } from '@mui/material';
import { Typography } from '@mui/material';
import { Button } from '@mui/material';
import ResultCard from '../ResultCard'
import DescCard from '../DescCard'
import AccordianResult from '../AccordianResult'
import { styled } from '@mui/material/styles';
import SettingsBackupRestoreIcon from '@mui/icons-material/SettingsBackupRestore';

/* Yelllow button */
const ColorButton = styled(Button)(({ theme }) => ({
    color: theme.palette.getContrastText('#ffb74d'),
    backgroundColor: '#ffb74d',
    '&:hover': {
      backgroundColor: '#fdd14d',
    },
  }));

/* This component displays the three different results, Desciption, Top predicitons and the next 9 predicitons*/  
const ResultsSection = ({executeSearchScroll, mainResult}) => {

    return (
        <>
        <Container maxWidth = 'xl'>
            <Box height="100vh" alignItems = "center" sx={{ py: '5rem'}}>
            <Grid container spacing={3} alignItems="flex-start" justifyContent="center">

                <Grid item xs={12}>
                    <Stack  direction="column" divider={<Divider orientation="horizontal" flexItem sx={{ my: 2 }}/>}>
                    
                    {/*Job Desciptions from MCF site*/}
                    <DescCard mainResult = {mainResult} originalText></DescCard>

                    {/*Predicted Results from API call*/}
                    <ResultCard mainResult = {mainResult} ></ResultCard>
                    
                    {/*Next 9 predictions arranged vertically, each prediction is mapped to an Accordian Result component*/}
                    <Grid container spacing={2} alignItems="center" justifyContent="center">
                        <Grid item xs={12}>
                            <Typography align="left">Next 9 predictions (in order) are:</Typography>
                        </Grid>
                        {mainResult.other_predictions?.map((nextpred, i) => ( // Mapping out all the predictions onto accordians      
                        <Grid item xs={12}>
                        <AccordianResult results = {nextpred} idx = {i}/>
                        </Grid>
                        ))}
                        <Grid item >
                            {/*Resets all entries, bring user back up to the first page*/}
                            <ColorButton endIcon={<SettingsBackupRestoreIcon />} onClick = {executeSearchScroll}>Click here to search for another entry</ColorButton>
                        </Grid>
                        <Grid item xs = {12}><div></div></Grid>
                    </Grid>
                    </Stack>
               
                </Grid>

            </Grid>
            </Box>
        </Container>

        </>
    )
}

export default ResultsSection
