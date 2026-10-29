const questionInput = document.getElementById("question");
const askButton = document.getElementById("ask-button");
const statusText = document.getElementById("status");
const pageUrlText = document.getElementById("page-url");

const answerSection = document.getElementById("answer-section");
const answerText = document.getElementById("answer");
const sourcesContainer = document.getElementById("sources");

let currentUrl = "";

async function updateCurrentPage() {
    try {
        const tab = await chrome.runtime.sendMessage({
            type: "GET_ACTIVE_TAB"
        });

        currentUrl = tab?.url || "";

        if (
            !currentUrl.startsWith("http://") &&
            !currentUrl.startsWith("https://")
        ) {
            currentUrl = "";

            pageUrlText.textContent =
                "This browser page cannot be processed.";

            return;
        }

        pageUrlText.textContent = currentUrl;

    } catch (error) {
        console.error("Tab detection error:", error);

        currentUrl = "";

        pageUrlText.textContent =
            "Unable to determine the current webpage.";
    }
}
chrome.tabs.onActivated.addListener(updateCurrentPage);

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.url || changeInfo.status === "complete") {
        updateCurrentPage();
    }
});

askButton.addEventListener("click", async () => {
    const question = questionInput.value.trim();

    if (!question) {
        statusText.textContent = "Please enter a question.";
        return;
    }

    await updateCurrentPage();

    if (!currentUrl) {
        statusText.textContent =
            "Open a normal webpage and try again.";
        return;
    }

    askButton.disabled = true;
    statusText.textContent =
        "Reading the webpage and generating an answer...";

    answerSection.hidden = true;
    answerText.textContent = "";
    sourcesContainer.replaceChildren();

    try {
        const response = await fetch(
            "http://127.0.0.1:8000/ask",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    url: currentUrl,
                    question: question
                })
            }
        );

        if (!response.ok) {
            throw new Error(
                `Backend returned HTTP ${response.status}`
            );
        }

        const data = await response.json();

        answerText.textContent =
            data.answer || "No answer was returned.";

        for (const source of data.sources || []) {
            try {
                const parsedUrl = new URL(source);

                if (
                    parsedUrl.protocol !== "http:" &&
                    parsedUrl.protocol !== "https:"
                ) {
                    continue;
                }

                const link = document.createElement("a");
                link.className = "source-link";
                link.href = parsedUrl.href;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                link.textContent = parsedUrl.href;

                sourcesContainer.appendChild(link);
            } catch {
                // Ignore invalid source URLs.
            }
        }

        answerSection.hidden = false;
        statusText.textContent = "Answer generated successfully.";
    } catch (error) {
        statusText.textContent =
            `Request failed: ${error.message}. ` +
            "Check that the WebRAG backend is running.";
    } finally {
        askButton.disabled = false;
    }
});

updateCurrentPage();