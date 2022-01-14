import React from 'react';
import Button from '@mui/material/Button';
import {
    SearchContainer, 
    SearchWrapper, 
    SearchRow, 
    TextWrapper,
    SearchBarWrapper,   
    Heading,
    Subtitle,
    BtnWrap
} from './SearchElements';
import TextField from '@mui/material/TextField';
import { styled } from '@mui/material/styles';


const ColorButton = styled(Button)(({ theme }) => ({
    color: theme.palette.getContrastText('#ffb74d'),
    backgroundColor: '#ffb74d',
    '&:hover': {
      backgroundColor: '#fdd14d',
    },
  }));

  
const ColorButtonWhite = styled(Button)(({ theme }) => ({
color: theme.palette.getContrastText('#ffb74d'),
backgroundColor: '#fff',
'&:hover': {
    backgroundColor: '#fdd14d',
},
}));

const SearchSection = ({ setMcfId, togglePress }) => {
    return (
        <>
            <SearchContainer>
                <SearchWrapper>
                    <SearchRow>
                        <TextWrapper>
                            <Heading>SSOC Autocoder</Heading>
                        </TextWrapper> 
                        <SearchBarWrapper>
                            <Subtitle>1. To start please enter the MCF Job ID below:</Subtitle>
                            <TextField fullWidth label="Input MCF Job ID here" onChange = {(event) => setMcfId(event.target.value)}></TextField>
                        </SearchBarWrapper>
                        <BtnWrap>
                            <ColorButton onClick = {togglePress}>Search</ColorButton>
                            <ColorButtonWhite variant="outlined">Feeling lucky</ColorButtonWhite>
                        </BtnWrap>
                    </SearchRow>
                </SearchWrapper>
            </SearchContainer>   
        </>
    )
}

export default SearchSection
