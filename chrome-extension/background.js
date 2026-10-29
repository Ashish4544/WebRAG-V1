chrome.runtime.onInstalled.addListener(() => {
    chrome.sidePanel.setPanelBehavior({
        openPanelOnActionClick: true
    });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "GET_ACTIVE_TAB") {
        chrome.tabs.query({
            active: true,
            lastFocusedWindow: true
        }).then((tabs) => {
            sendResponse({
                url: tabs[0]?.url || "",
                tabId: tabs[0]?.id ?? null
            });
        }).catch((error) => {
            console.error(error);
            sendResponse({ url: "", tabId: null });
        });

        return true;
    }
});