/* Importing packages */
import React from 'react';
import { Button } from '@mui/material';
import { Typography } from '@mui/material';
import { TextField} from '@mui/material';
import { Box } from '@mui/material';
import { Grid } from '@mui/material';
import { Container } from '@mui/material';
import { Link } from '@mui/material';
import { styled } from '@mui/material/styles';
import CustInfoIcon from '../custInfoIcon';

/* Yellow button */
const ColorButton = styled(Button)(({ theme }) => ({
    color: theme.palette.getContrastText('#ffb74d'),
    backgroundColor: '#ffb74d',
    '&:hover': {
      backgroundColor: '#fdd14d',
    },
  }));

/* White button */
const ColorButtonWhite = styled(Button)(({ theme }) => ({
    color: theme.palette.getContrastText('#ffb74d'),
    backgroundColor: '#fff',
    '&:hover': {
        backgroundColor: '#fdd14d',
    },
    }));


/* Main page for users to enter their query into a search bar */
const SearchSection = ({ searchBarRef, mcfUrl, setMcfUrl, togglePress, toggleDefault}) => {
    return (
        <>
             <Container>
             <Box height="60vh" alignItems = "center" sx={{ pt: "30%"}}>
                
                <Grid container rowSpacing={3} direction = "row" >
                    <Grid item xs = {12} >
                        <Typography variant="h3" gutterBottom component="div" align = "center">
                            <strong>SSOC Autocoder</strong>
                        </Typography>
                    </Grid>

                    <Grid item xs = {12} >
                        <Typography>Head over to the <Link target="_blank" href = "https://www.mycareersfuture.gov.sg/" underline="always" >MyCareersFuture</Link> website. Choose a job advertisment and paste the URL below:
                 
                        <CustInfoIcon />
  
                        
                        </Typography>
                        
                    </Grid>
                    
                    <Grid item xs = {12} >
                        {/* Input field for users to enter MCF job description URL */}
                        <TextField inputProps={{ inputMode: 'none' }} value = {mcfUrl} inputRef = {searchBarRef} fullWidth label="Paste MCF job ad URL here" onChange = {(event) => setMcfUrl(event.target.value)}></TextField>
                    </Grid>
                    
                    <Grid item xs = {12} >
                        <Grid container spacing={2} align = "center">
                        
                            <Grid item xs = {6} >
                            <ColorButton onClick = {togglePress}>Search</ColorButton>
                            </Grid>
                            <Grid item xs = {6} >
                            <ColorButtonWhite onClick = {toggleDefault} variant="outlined">Feeling lucky</ColorButtonWhite>
                            </Grid>
            
                        </Grid>
                    </Grid>

                </Grid>
            
            </Box>
            </Container>
        </>
    )
}

export default SearchSection
