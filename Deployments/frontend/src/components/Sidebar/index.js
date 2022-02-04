import React from 'react'
import {
    SidebarContainer,
    Icon,
    CloseIcon,
    SidebarWrapper,
    SidebarMenu,
    SidebarLink,
    SideBtmWrap,
    SidebarRoute
} from './SidebarElements'

const Sidebar = ({ isOpen, toggle }) => {
    return (
        <SidebarContainer isOpen = {isOpen} onClick = {toggle}>
            <Icon onClick={toggle}>
                <CloseIcon />
            </Icon>
            <SidebarWrapper>
                <SidebarMenu>
                    <SidebarLink to = "search">Search</SidebarLink>
                    <SidebarLink to = "about">About</SidebarLink>
                    <SidebarLink to = "api">API</SidebarLink>
                    <SidebarLink to = "contact-us">Contact Us</SidebarLink>
                </SidebarMenu>
                <SideBtmWrap>
                    <SidebarRoute to="/load-api">Load API Key</SidebarRoute>
                </SideBtmWrap>
            </SidebarWrapper>
        </SidebarContainer>
    )
}

export default Sidebar
