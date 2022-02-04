/* Importing packages */
import React from 'react'
import { Typography } from '@mui/material';
import { Box } from '@mui/material';
import { Grid } from '@mui/material';
import { Container } from '@mui/material';
import { CircularProgress } from '@mui/material';

/* Wating screen before everything is loaded */
const LoadingSection = () => {
    return (
        <Container>
        <Box height="100vh" alignItems = "center" sx={{ pt: "35%"}}>

            <Grid container rowSpacing={3} direction = "row" >

                <Grid item xs = {12} align = "center">
                    <CircularProgress size = '10rem'/>
                </Grid>

                <Grid item xs = {12} align = "center">
                <Typography variant="h5">Give us a moment to generate the predictions...</Typography>
                </Grid>
            </Grid>
            
        
       </Box>
       </Container>
    )
}

export default LoadingSection
